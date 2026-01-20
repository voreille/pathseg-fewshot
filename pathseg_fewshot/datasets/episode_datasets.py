from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pathseg_fewshot.datasets.episode_sampler import (
    EpisodeRefs,
    EpisodeSampler,
)


# -----------------------------
# Helpers: IO
# -----------------------------
def load_image_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_ids(path: Path) -> np.ndarray:
    """
    Loads a label mask as integer IDs.
    Assumes your preprocessing saved masks as 'L' (uint8) or 'P' palette (indices).
    """
    with Image.open(path) as im:
        if im.mode == "P":
            return np.array(im, dtype=np.int32)
        if im.mode == "L":
            return np.array(im, dtype=np.int32)
        raise ValueError(f"Unsupported mask mode {im.mode} for {path}")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL RGB to float tensor [C,H,W] in [0,1].
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,C]
    arr = np.transpose(arr, (2, 0, 1))  # [C,H,W]
    return torch.from_numpy(arr)


def sample_crop_center_on_class(
    mask: np.ndarray,
    class_id: int,
    crop_size: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """
    Choose a crop top-left (y0,x0) such that crop likely contains pixels of `class_id`.
    Strategy: sample a pixel coordinate from that class, then center crop around it.

    Falls back to uniform crop if class pixels not found.
    """
    H, W = mask.shape
    cs = crop_size
    if H < cs or W < cs:
        # caller should handle resizing/padding; keep simple: clamp later
        pass

    ys, xs = np.where(mask == class_id)
    if ys.size == 0:
        # fallback: random crop
        y0 = rng.randint(0, max(0, H - cs))
        x0 = rng.randint(0, max(0, W - cs))
        return y0, x0

    idx = rng.randrange(ys.size)
    cy, cx = int(ys[idx]), int(xs[idx])

    y0 = cy - cs // 2
    x0 = cx - cs // 2

    y0 = max(0, min(y0, max(0, H - cs)))
    x0 = max(0, min(x0, max(0, W - cs)))
    return y0, x0


def crop_pair(
    img: Image.Image,
    mask: np.ndarray,
    y0: int,
    x0: int,
    crop_size: int,
) -> Tuple[Image.Image, np.ndarray]:
    cs = crop_size
    img_c = img.crop((x0, y0, x0 + cs, y0 + cs))
    mask_c = mask[y0 : y0 + cs, x0 : x0 + cs]
    return img_c, mask_c


# -----------------------------
# Transforms
# -----------------------------
class PairedTransform:
    """
    Applies class-conditioned random crop and optional augmentation to an (image, mask) pair.
    You can plug your own augmentation function that takes (PIL_img, np_mask) and returns same.
    """

    def __init__(
        self,
        crop_size: int = 448,
        augmentation: Optional[
            Callable[
                [Image.Image, np.ndarray, random.Random], Tuple[Image.Image, np.ndarray]
            ]
        ] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.crop_size = int(crop_size)
        self.augmentation = augmentation
        self._base_seed = seed

    def __call__(
        self,
        img: Image.Image,
        mask: np.ndarray,
        *,
        class_id: int,
        rng: random.Random,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If images can be smaller than crop, you should decide: pad or resize.
        # Minimal approach: raise early (forces you to preprocess consistently).
        H, W = mask.shape
        if H < self.crop_size or W < self.crop_size:
            raise ValueError(
                f"Image/mask smaller than crop_size={self.crop_size}: got {(H, W)}"
            )

        y0, x0 = sample_crop_center_on_class(mask, class_id, self.crop_size, rng)
        img_c, mask_c = crop_pair(img, mask, y0, x0, self.crop_size)

        if self.augmentation is not None:
            img_c, mask_c = self.augmentation(img_c, mask_c, rng)

        x = pil_to_tensor(img_c)  # [C,H,W], float
        y = torch.from_numpy(mask_c.astype(np.int64))  # [H,W], long
        return x, y


class OnTheFlyEpisodeDataset(Dataset):
    """
    TRAIN: on-the-fly episodes sampled deterministically from idx.
    """

    def __init__(
        self,
        root_dir: Path,
        *,
        sampler: EpisodeSampler,
        episodes_per_epoch: int,
        transform: PairedTransform,
        base_seed: int = 0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.sampler = sampler
        self.episodes_per_epoch = int(episodes_per_epoch)
        if self.episodes_per_epoch <= 0:
            raise ValueError("episodes_per_epoch must be > 0")
        self.transform = transform
        self.base_seed = int(base_seed)

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode_seed = (self.base_seed * 1_000_000) + int(idx)
        episode = self.sampler.sample_episode(seed=episode_seed)
        if episode is None:
            raise RuntimeError(
                "Sampler returned None. Did you set with_replacement=False for training?"
            )

        rng = random.Random(episode_seed)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for ref in episode.support:
            img = load_image_rgb(self.root_dir / ref.dataset_id / ref.image_relpath)
            mask = load_mask_ids(self.root_dir / ref.dataset_id / ref.mask_relpath)
            x, y = self.transform(
                img, mask, class_id=ref.class_id, rng=rng, crop=ref.crop
            )
            support_x.append(x)
            support_y.append(y)

        for ref in episode.query:
            img = load_image_rgb(self.root_dir / ref.dataset_id / ref.image_relpath)
            mask = load_mask_ids(self.root_dir / ref.dataset_id / ref.mask_relpath)
            x, y = self.transform(
                img, mask, class_id=ref.class_id, rng=rng, crop=ref.crop
            )
            query_x.append(x)
            query_y.append(y)

        return {
            "dataset_id": episode.dataset_id,
            "class_ids": torch.tensor(episode.class_ids, dtype=torch.long),
            "support_images": torch.stack(support_x, dim=0),
            "support_masks": torch.stack(support_y, dim=0),
            "query_images": torch.stack(query_x, dim=0),
            "query_masks": torch.stack(query_y, dim=0),
            "seed": episode_seed,
        }


class EpisodeListDataset(Dataset):
    """
    VAL/TEST: consume a fixed serialized list of episodes.
    """

    def __init__(
        self,
        root_dir: Path,
        *,
        episodes: List[EpisodeRefs],
        transform: PairedTransform,
        base_seed: int = 0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.episodes = episodes
        self.transform = transform
        self.base_seed = int(base_seed)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode = self.episodes[int(idx)]
        episode_seed = (self.base_seed * 1_000_000) + int(idx)
        rng = random.Random(episode_seed)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for ref in episode.support:
            img = load_image_rgb(self.root_dir / ref.dataset_id / ref.image_relpath)
            mask = load_mask_ids(self.root_dir / ref.dataset_id / ref.mask_relpath)
            x, y = self.transform(
                img, mask, class_id=ref.class_id, rng=rng, crop=ref.crop
            )
            support_x.append(x)
            support_y.append(y)

        for ref in episode.query:
            img = load_image_rgb(self.root_dir / ref.dataset_id / ref.image_relpath)
            mask = load_mask_ids(self.root_dir / ref.dataset_id / ref.mask_relpath)
            x, y = self.transform(
                img, mask, class_id=ref.class_id, rng=rng, crop=ref.crop
            )
            query_x.append(x)
            query_y.append(y)

        return {
            "dataset_id": episode.dataset_id,
            "class_ids": torch.tensor(episode.class_ids, dtype=torch.long),
            "support_images": torch.stack(support_x, dim=0),
            "support_masks": torch.stack(support_y, dim=0),
            "query_images": torch.stack(query_x, dim=0),
            "query_masks": torch.stack(query_y, dim=0),
            "seed": episode_seed,
        }
