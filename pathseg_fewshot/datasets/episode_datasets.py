from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

from pathseg_fewshot.datasets.episode import EpisodeRefs, SampleRef
from pathseg_fewshot.datasets.episode_sampler import EpisodeSamplerBase


def load_image(path: Path) -> torch.Tensor:
    # returns uint8 [3,H,W]
    return read_image(
        str(path),
        mode=ImageReadMode.RGB,
    )


def load_mask(path: Path) -> torch.Tensor:
    # returns uint8 [1,H,W] → squeeze to [H,W]
    mask = read_image(
        str(path),
        mode=ImageReadMode.GRAY,
    )
    return mask.squeeze(0).to(torch.long)


class EpisodeDatasetBase(Dataset):
    """
    Episode dataset that returns *semantic* per-pixel labels (no instance masks).

    Each sample target is:
      target = {"mask": tv_tensors.Mask[H,W] (long)}
    """

    def __init__(
        self,
        root_dir: Path,
        *,
        sampler: EpisodeSamplerBase,
        episodes_per_epoch: int,
        transform: Optional[torch.nn.Module],
        base_seed: int = 0,
        ignore_idx: int = 255,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.sampler = sampler
        self.episodes_per_epoch = int(episodes_per_epoch)
        if self.episodes_per_epoch <= 0:
            raise ValueError("episodes_per_epoch must be > 0")

        self.transform = transform
        self.base_seed = int(base_seed)
        self.ignore_idx = int(ignore_idx)

    def load_img_mask(
        self,
        *,
        ref: SampleRef,
        data_dir: Path,
        class_ids: List[int],
    ) -> Tuple[tv_tensors.Image, tv_tensors.Mask]:
        image = load_image(data_dir / ref.image_relpath)
        mask = load_mask(data_dir / ref.mask_relpath)

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        mask = mask.long()
        ignore = int(self.ignore_idx)  # e.g. 255

        # Default: everything becomes background (0)
        remapped = torch.zeros_like(mask, dtype=torch.long)

        # Keep ignore pixels as ignore
        remapped[mask == ignore] = ignore

        # Map selected classes to 1..K (bg is 0)
        for new_idx, cid in enumerate(class_ids, start=1):
            remapped[mask == cid] = new_idx

        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(remapped)

        target = {"mask": mask}
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target["mask"]

    def get_episode(self, idx: int):
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode, episode_seed = self.get_episode(idx)
        if episode is None:
            raise RuntimeError(
                "Sampler returned None. Did you set with_replacement=False for training?"
            )

        support_x, support_y = [], []
        query_x, query_y = [], []
        data_dir = self.root_dir / episode.dataset_dir
        class_ids = episode.class_ids

        for ref in episode.support:
            img, target = self.load_img_mask(
                ref=ref, data_dir=data_dir, class_ids=class_ids
            )
            support_x.append(img)
            support_y.append(target)  # contains {"mask": ...}

        for ref in episode.query:
            img, target = self.load_img_mask(
                ref=ref, data_dir=data_dir, class_ids=class_ids
            )
            query_x.append(img)
            query_y.append(target)

        return {
            "dataset_id": episode.dataset_id,
            "class_ids": torch.tensor(class_ids, dtype=torch.long),
            "support_images": support_x,
            "support_masks": support_y,  # renamed (semantic target dicts)
            "query_images": query_x,
            "query_masks": query_y,  # renamed
            "seed": episode_seed,
        }


class OnTheFlyEpisodeDataset(EpisodeDatasetBase):
    """
    TRAIN: on-the-fly episodes sampled deterministically from idx.
    """

    def __init__(
        self,
        root_dir: Path,
        *,
        sampler: EpisodeSamplerBase,
        episodes_per_epoch: int,
        transform: Optional[torch.nn.Module],
        base_seed: int = 0,
        ignore_idx: int = 255,
    ) -> None:
        super().__init__(
            root_dir,
            sampler=sampler,
            episodes_per_epoch=episodes_per_epoch,
            transform=transform,
            base_seed=base_seed,
            ignore_idx=ignore_idx,
        )

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def get_episode(self, idx: int) -> Tuple[EpisodeRefs, int]:
        episode_seed = (self.base_seed * 1_000_000) + int(idx)
        return self.sampler.sample_episode(seed=episode_seed), episode_seed


class EpisodeListDataset(EpisodeDatasetBase):
    """
    VAL/TEST: consume a fixed serialized list of episodes.
    """

    def __init__(
        self,
        root_dir: Path,
        *,
        episodes: List[EpisodeRefs],
        transform: Optional[torch.nn.Module] = None,
        base_seed: int = 0,
        ignore_idx: int = 255,
    ) -> None:
        # sampler/episodes_per_epoch aren't used here, but base class expects them.
        # We'll pass a dummy sampler and episodes_per_epoch; get_episode is overridden.
        self.episodes = episodes
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.base_seed = int(base_seed)
        self.ignore_idx = int(ignore_idx)

    def __len__(self) -> int:
        return len(self.episodes)

    def get_episode(self, idx: int) -> Tuple[EpisodeRefs, int]:
        episode_seed = (self.base_seed * 1_000_000) + int(idx)
        return self.episodes[idx], episode_seed

    # Reuse EpisodeDatasetBase.__getitem__ by inheriting it;
    # it calls self.get_episode() which we override above.
