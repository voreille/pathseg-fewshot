from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import numpy as np
import openslide
import pandas as pd
import tifffile as tiff
from PIL import Image, ImageDraw
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    log_file = output_dir / "tiling.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers (important for CLI reruns)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging initialized.")
    logger.info(f"Log file: {log_file}")


# ----------------------------
# Utilities
# ----------------------------


def get_mpp(image: openslide.OpenSlide) -> Tuple[float, float]:
    mpp_x = image.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = image.properties.get(openslide.PROPERTY_NAME_MPP_Y)

    if mpp_x is None or mpp_y is None:
        logger.warning("MPP not found. Defaulting to 0.5 µm/px")
        return 0.5, 0.5

    return float(mpp_x), float(mpp_y)


def mpp_to_nominal_magnification(mpp: float) -> float:
    # Standard approximation
    return 10.0 / mpp


def select_level_for_magnification(
    wsi: openslide.OpenSlide,
    target_mag: float,
) -> int:
    mpp_x, _ = get_mpp(wsi)
    best_level = 0
    smallest_diff = float("inf")

    for level, ds in enumerate(wsi.level_downsamples):
        level_mpp = mpp_x * ds
        level_mag = mpp_to_nominal_magnification(level_mpp)
        diff = abs(level_mag - target_mag)

        if diff < smallest_diff:
            smallest_diff = diff
            best_level = level

    logger.info(f"Selected level {best_level} for target {target_mag}x")

    return best_level


# ----------------------------
# Tiling
# ----------------------------


def tile_wsi(
    wsi_path: Path,
    mask_path: Path,
    output_img_dir: Path,
    output_mask_dir: Path,
    dataset_id: str,
    roi_size: int,
    roi_stride: int,
    target_mag: int,
    labelled_ratio_threshold: float = 0.8,
    min_label_pixels: int | None = None,
    save_overview: bool = False,
    overview_dir: Path | None = None,
    overview_max_size: int = 2000,
    start_short_candidates: int = 8,  # number of shifts to try within one ROI
) -> List[Dict[str, Any]]:
    wsi = openslide.OpenSlide(str(wsi_path))

    mpp_x0, mpp_y0 = get_mpp(wsi)
    level = select_level_for_magnification(wsi, target_mag)
    ds = float(wsi.level_downsamples[level])
    w, h = wsi.level_dimensions[level]

    # Read mask at the same level with tifffile
    with tiff.TiffFile(mask_path) as tf:
        mask = tf.pages[level].asarray()

    if mask.ndim == 3:
        logger.info("Mask is RGB, using channel 0.")
        mask = mask[..., 0]

    if mask.shape[:2] != (h, w):
        raise ValueError(
            f"Mask/WSI mismatch at level {level}: "
            f"WSI={w}x{h}, Mask={mask.shape[1]}x{mask.shape[0]}"
        )

    if min_label_pixels is None:
        min_label_pixels = int(0.05 * roi_size * roi_size)

    # bbox of non-zero
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        logger.warning(f"No labels in mask: {mask_path.name}")
        return []

    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    ymin, ymax = int(ys.min()), int(ys.max()) + 1

    bw = xmax - xmin
    bh = ymax - ymin
    long_is_x = bw >= bh

    def clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(v, hi))

    # assign long/short coords
    if long_is_x:
        bmin_long, bmax_long, dim_long = xmin, xmax, w
        bmin_short, bmax_short, dim_short = ymin, ymax, h
    else:
        bmin_long, bmax_long, dim_long = ymin, ymax, h
        bmin_short, bmax_short, dim_short = xmin, xmax, w

    # snap long window to multiple of roi_size (ceil)
    span_long = bmax_long - bmin_long
    n_long = int(np.ceil(span_long / roi_size))
    snapped_long = max(roi_size, n_long * roi_size)
    pad_long = snapped_long - span_long

    start_long = bmin_long - pad_long // 2
    start_long = clamp(start_long, 0, dim_long - snapped_long)
    end_long = start_long + snapped_long

    # candidate short starts: shift around bbox start by up to ~1 ROI
    # (this is what makes the grid "dynamic")
    if start_short_candidates <= 1:
        shifts = [0]
    else:
        step = max(1, roi_size // start_short_candidates)
        shifts = list(range(0, roi_size, step))[:start_short_candidates]

    all_rois: List[Tuple[int, int]] = []
    selected_rois: List[Tuple[int, int]] = []
    rows: List[Dict[str, Any]] = []

    # iterate strips along long axis
    for p_long in range(start_long, end_long - roi_size + 1, roi_stride):
        # choose best start_short FOR THIS STRIP
        best_start_short = None
        best_score = -1

        for sh in shifts:
            # try starting a little before bbox start, shifted by sh
            cand = bmin_short - sh
            cand = clamp(cand, 0, dim_short - roi_size)

            score = 0
            # score only within this strip
            for p_short in range(cand, dim_short - roi_size + 1, roi_stride):
                if long_is_x:
                    patch = mask[
                        p_short : p_short + roi_size, p_long : p_long + roi_size
                    ]
                else:
                    patch = mask[
                        p_long : p_long + roi_size, p_short : p_short + roi_size
                    ]
                score += int(np.count_nonzero(patch))

            if score > best_score:
                best_score = score
                best_start_short = cand

        assert best_start_short is not None

        # tile this strip using the chosen start_short
        for p_short in range(best_start_short, dim_short - roi_size + 1, roi_stride):
            if long_is_x:
                x, y = p_long, p_short
            else:
                x, y = p_short, p_long

            all_rois.append((x, y))

            tile_mask = mask[y : y + roi_size, x : x + roi_size]
            if tile_mask.shape != (roi_size, roi_size):
                continue

            label_pixels = int(np.count_nonzero(tile_mask))
            ratio = label_pixels / tile_mask.size

            if ratio < labelled_ratio_threshold and label_pixels < min_label_pixels:
                continue

            tile_img = wsi.read_region((x, y), level, (roi_size, roi_size)).convert(
                "RGB"
            )

            img_name = f"{wsi_path.stem}_L{level}_{x}_{y}.png"
            msk_name = f"{wsi_path.stem}_L{level}_{x}_{y}.png"

            tile_img.save(output_img_dir / img_name)
            Image.fromarray(tile_mask.astype(np.uint8)).save(output_mask_dir / msk_name)

            selected_rois.append((x, y))

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_id": wsi_path.stem,
                    "group": wsi_path.stem,
                    "image_relpath": str(Path("images") / img_name),
                    "mask_relpath": str(Path("masks_semantic") / msk_name),
                    "width": roi_size,
                    "height": roi_size,
                    "mpp_x": mpp_x0 * ds,
                    "mpp_y": mpp_y0 * ds,
                    "magnification": target_mag,
                    "level": level,
                    "level_downsample": ds,
                    "labelled_ratio": ratio,
                    "label_pixels": label_pixels,
                }
            )

    logger.info(
        f"{wsi_path.stem}: total={len(all_rois)} kept={len(selected_rois)} "
        f"(thr={labelled_ratio_threshold}, min_pixels={min_label_pixels})"
    )

    # keep your existing overview code; it will now show strip-specific shifts
    # (all_rois vs selected_rois)

    return rows


def tile_wsi_grid(
    wsi_path: Path,
    mask_path: Path,
    output_img_dir: Path,
    output_mask_dir: Path,
    dataset_id: str,
    tile_size: int,
    stride: int,
    target_mag: int,
    labelled_ratio_threshold: float = 0.8,
    save_overview: bool = False,
    overview_dir: Path | None = None,
    overview_max_size: int = 2000,
) -> List[Dict[str, Any]]:
    wsi = openslide.OpenSlide(str(wsi_path))

    mpp_x, mpp_y = get_mpp(wsi)
    level = select_level_for_magnification(wsi, target_mag)
    level_downsample = wsi.level_downsamples[level]

    w, h = wsi.level_dimensions[level]
    rows = []

    with tiff.TiffFile(mask_path) as tf:
        mask = tf.pages[level].asarray()

    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.shape[0] != h or mask.shape[1] != w:
        raise ValueError("Mask and WSI dimensions do not match.")

    all_tiles = []
    selected_tiles = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            # --- Extract WSI tile ---
            all_tiles.append((x, y))
            tile_mask = mask[y : y + tile_size, x : x + tile_size]
            if tile_mask.shape != (tile_size, tile_size):
                continue
            if np.all(tile_mask == 0):
                continue

            if np.sum(tile_mask > 0) / tile_mask.size < labelled_ratio_threshold:
                continue

            tile_img = wsi.read_region(
                (x, y),
                level,
                (tile_size, tile_size),
            ).convert("RGB")

            if tile_img.size != (tile_size, tile_size):
                logger.warning(
                    f"Resampling tile from {tile_img.size} to {(tile_size, tile_size)}"
                )
                tile_img = tile_img.resize((tile_size, tile_size), Image.LANCZOS)

            img_name = f"{wsi_path.stem}_{x}_{y}.png"
            msk_name = f"{wsi_path.stem}_{x}_{y}.png"

            img_out = output_img_dir / img_name
            msk_out = output_mask_dir / msk_name

            tile_img.save(img_out)
            Image.fromarray(tile_mask.astype(np.uint8)).save(msk_out)

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_id": wsi_path.stem,
                    "group": wsi_path.stem,
                    "image_relpath": str(Path("images") / img_name),
                    "mask_relpath": str(Path("masks_semantic") / msk_name),
                    "width": tile_size,
                    "height": tile_size,
                    "mpp_x": mpp_x,
                    "mpp_y": mpp_y,
                    "magnification": target_mag,
                }
            )
            selected_tiles.append((x, y))

    if save_overview and overview_dir is not None:
        overview_dir.mkdir(exist_ok=True)

        # Read lowest resolution level for thumbnail
        thumb_level = wsi.level_count - 1
        thumb_w, thumb_h = wsi.level_dimensions[thumb_level]

        thumbnail = wsi.read_region(
            (0, 0),
            thumb_level,
            (thumb_w, thumb_h),
        ).convert("RGB")

        # Resize if too large
        scale_factor = 1.0
        max_dim = max(thumbnail.size)

        if max_dim > overview_max_size:
            scale_factor = overview_max_size / max_dim
            new_size = (
                int(thumbnail.width * scale_factor),
                int(thumbnail.height * scale_factor),
            )
            thumbnail = thumbnail.resize(new_size, Image.BILINEAR)

        draw = ImageDraw.Draw(thumbnail)

        level_downsample = wsi.level_downsamples[thumb_level]

        # Draw grid (light gray)
        for x, y in all_tiles:
            x_thumb = int((x / level_downsample) * scale_factor)
            y_thumb = int((y / level_downsample) * scale_factor)
            size_thumb = int((tile_size / level_downsample) * scale_factor)

            draw.rectangle(
                [x_thumb, y_thumb, x_thumb + size_thumb, y_thumb + size_thumb],
                outline=(180, 180, 180),
                width=1,
            )

        # Draw selected tiles (green)
        for x, y in selected_tiles:
            x_thumb = int((x / level_downsample) * scale_factor)
            y_thumb = int((y / level_downsample) * scale_factor)
            size_thumb = int((tile_size / level_downsample) * scale_factor)

            draw.rectangle(
                [x_thumb, y_thumb, x_thumb + size_thumb, y_thumb + size_thumb],
                outline=(0, 255, 0),
                width=2,
            )

        overview_path = overview_dir / f"{wsi_path.stem}_overview.png"
        thumbnail.save(overview_path)

    return rows


# ----------------------------
# CLI
# ----------------------------


@click.command()
@click.option(
    "--raw-data-dir", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--dataset-id", default="panda", show_default=True)
@click.option("--tile-size", default=1120, show_default=True)
@click.option("--tile-stride", default=1120, show_default=True)
@click.option("--target-magnification", default=20, show_default=True)
@click.option("--labelled-ratio-threshold", default=0.8, show_default=True)
@click.option("--save-overview", is_flag=True, default=False, show_default=True)
def main(
    raw_data_dir: Path,
    output_dir: Path,
    dataset_id: str,
    tile_size: int,
    tile_stride: int,
    target_magnification: int,
    labelled_ratio_threshold: float,
    save_overview: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks_semantic"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    if save_overview:
        overview_dir = output_dir / "overviews"
        overview_dir.mkdir(exist_ok=True)
    else:
        overview_dir = None

    train_df = pd.read_csv(raw_data_dir / "train.csv")
    train_df = train_df[train_df["data_provider"] == "radboud"]

    rows = []

    for _, r in tqdm(train_df.iterrows(), total=len(train_df)):
        wsi_id = r["image_id"]

        wsi_path = raw_data_dir / "train_images" / f"{wsi_id}.tiff"
        mask_path = raw_data_dir / "train_label_masks" / f"{wsi_id}_mask.tiff"

        if not wsi_path.exists() or not mask_path.exists():
            logger.warning(f"Missing files for {wsi_id}")
            continue

        rows.extend(
            tile_wsi(
                wsi_path=wsi_path,
                mask_path=mask_path,
                output_img_dir=images_dir,
                output_mask_dir=masks_dir,
                dataset_id=dataset_id,
                tile_size=tile_size,
                tile_stride=tile_stride,
                target_mag=target_magnification,
                labelled_ratio_threshold=labelled_ratio_threshold,
                save_overview=save_overview,
                overview_dir=overview_dir if save_overview else None,
            )
        )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    logger.info("Tiling complete.")


if __name__ == "__main__":
    main()
