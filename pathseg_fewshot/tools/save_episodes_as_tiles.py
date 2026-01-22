from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from pathseg_fewshot.datasets.episode import EpisodeRefs, SampleRef, load_episodes_json


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _mask_to_binary(mask: Image.Image) -> np.ndarray:
    """
    Convert mask to a boolean foreground mask.
    Works for:
      - L/P palette masks
      - 8-bit grayscale masks
      - RGB masks (any channel > 0)
      - 16-bit masks
    """
    arr = np.array(mask)
    if arr.ndim == 2:
        return arr > 0
    # RGB / RGBA
    return np.any(arr[..., :3] > 0, axis=-1)


def _color_from_class_id(class_id: int) -> Tuple[int, int, int]:
    """
    Deterministic vivid-ish color from class_id (no external deps).
    """
    # Simple hashing to get stable, non-dark colors
    x = (class_id * 2654435761) & 0xFFFFFFFF
    r = 64 + (x & 0x7F)
    g = 64 + ((x >> 8) & 0x7F)
    b = 64 + ((x >> 16) & 0x7F)
    return int(r), int(g), int(b)


def _overlay_mask_on_image(
    img_rgb: Image.Image,
    mask: Image.Image,
    class_id: int,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Overlay a binary foreground mask on top of RGB image with a class color.
    alpha blends only where mask is foreground.
    """
    img_rgb = _ensure_rgb(img_rgb)

    img_arr = np.asarray(img_rgb).astype(np.float32)
    fg = _mask_to_binary(mask)

    color = np.array(_color_from_class_id(class_id), dtype=np.float32)

    out = img_arr.copy()
    out[fg] = (1.0 - alpha) * out[fg] + alpha * color
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _load_and_crop(
    root_dir: Path,
    r: SampleRef,
) -> Tuple[Image.Image, Image.Image]:
    """
    Loads image & mask from disk and applies r.crop if not None.
    """
    img_path = root_dir / r.dataset_id / r.image_relpath
    mask_path = root_dir / r.dataset_id / r.mask_relpath

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    if r.crop is not None:
        # PIL crop box is (left, top, right, bottom)
        img = img.crop(r.crop)
        mask = mask.crop(r.crop)

    return img, mask


def _save_one_ref(
    root_dir: Path,
    out_dir: Path,
    r: "SampleRef",
    prefix: str,
    alpha: float = 0.45,
) -> None:
    """
    Saves image, mask, and overlay to out_dir.
    """
    img, mask = _load_and_crop(root_dir, r)

    # Decide filenames
    # include class id to make browsing easier
    stem = f"{prefix}_class{int(r.class_id):03d}_sample{r.sample_id}"

    img_out = out_dir / f"{stem}_image.png"
    mask_out = out_dir / f"{stem}_mask.png"
    overlay_out = out_dir / f"{stem}_overlay.png"

    # Save raw image (force RGB for consistent viewing)
    _ensure_rgb(img).save(img_out)

    # Save mask as-is (keeps palette etc). If you prefer grayscale, convert("L").
    mask.save(mask_out)

    # Save overlay
    overlay = _overlay_mask_on_image(img, mask, class_id=int(r.class_id), alpha=alpha)
    overlay.save(overlay_out)


def save_episodes_as_tiles(
    root_dir: Path,
    output_dir: Path,
    episodes: List[EpisodeRefs],
    alpha: float = 0.45,
) -> None:
    """
    For each episode e:
      output_dir/episode_XXXX/support/...
      output_dir/episode_XXXX/query/...
    Save image, mask, overlay for each SampleRef.

    Also writes output_dir/episodes.json for traceability.
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the episode definition JSON next to the visualization
    # (same structure as your save_episodes_json)
    payload = []
    for e in episodes:
        payload.append(
            {
                "dataset_id": e.dataset_id,
                "class_ids": list(map(int, e.class_ids)),
                "support": [
                    {
                        "dataset_id": r.dataset_id,
                        "sample_id": r.sample_id,
                        "image_relpath": r.image_relpath,
                        "mask_relpath": r.mask_relpath,
                        "class_id": int(r.class_id),
                        "crop": r.crop,
                    }
                    for r in e.support
                ],
                "query": [
                    {
                        "dataset_id": r.dataset_id,
                        "sample_id": r.sample_id,
                        "image_relpath": r.image_relpath,
                        "mask_relpath": r.mask_relpath,
                        "class_id": int(r.class_id),
                        "crop": r.crop,
                    }
                    for r in e.query
                ],
            }
        )

    with (output_dir / "episodes.json").open("w") as f:
        json.dump(payload, f, indent=2)

    # Now render episodes
    for e_idx, e in enumerate(tqdm(episodes, desc="Rendering episodes")):
        episode_dir = output_dir / f"episode_{e_idx:04d}"
        support_dir = episode_dir / "support"
        query_dir = episode_dir / "query"
        support_dir.mkdir(parents=True, exist_ok=True)
        query_dir.mkdir(parents=True, exist_ok=True)

        # Support
        for i, r in enumerate(e.support):
            _save_one_ref(
                root_dir=root_dir,
                out_dir=support_dir,
                r=r,
                prefix=f"{i:03d}",
                alpha=alpha,
            )

        # Query
        for i, r in enumerate(e.query):
            _save_one_ref(
                root_dir=root_dir,
                out_dir=query_dir,
                r=r,
                prefix=f"{i:03d}",
                alpha=alpha,
            )


if __name__ == "__main__":
    episodes_json = Path(
        "/home/val/workspaces/pathseg-fewshot/data/splits/scenario_1/val_episodes.json"
    )
    episodes = load_episodes_json(episodes_json) 
    save_episodes_as_tiles(
        root_dir=Path("/home/val/workspaces/pathseg-fewshot/data/fss/"),
        output_dir=Path(
            "/home/val/workspaces/pathseg-fewshot/data/visualizations/val_episodes_tiles/"
        ),
        episodes=episodes,
        alpha=0.45,
    )
