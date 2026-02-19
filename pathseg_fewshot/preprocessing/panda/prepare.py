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
from scipy import ndimage as ndi
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


def _save_overview(
    *,
    wsi: openslide.OpenSlide,
    mask_path: Path,
    used_rois_lvl0: List[Tuple[int, int]],
    kept_rois_lvl0: List[Tuple[int, int]],
    strip_anchors_lvl0: List[Tuple[int, int]],
    roi_size_at_target: int,
    target_level: int,
    out_path: Path,
    overview_max_size: int = 2000,
) -> None:
    """Save thumbnail with (1) green binary mask overlay, (2) used grid, (3) kept tiles, (4) strip anchors."""
    thumb_level = wsi.level_count - 1
    tw, th = wsi.level_dimensions[thumb_level]
    thumb = wsi.read_region((0, 0), thumb_level, (tw, th)).convert("RGB")

    # downsample to max size
    scale = 1.0
    if max(thumb.size) > overview_max_size:
        scale = overview_max_size / max(thumb.size)
        thumb = thumb.resize(
            (int(thumb.width * scale), int(thumb.height * scale)), Image.BILINEAR
        )

    ds_thumb = float(wsi.level_downsamples[thumb_level])
    ds_target = float(wsi.level_downsamples[target_level])

    def lvl0_to_thumb(v: int) -> int:
        return int((v / ds_thumb) * scale)

    # ROI size in level0 -> thumb
    roi_size_lvl0 = int(round(roi_size_at_target * ds_target))
    roi_size_thumb = max(1, int((roi_size_lvl0 / ds_thumb) * scale))

    # --- binary mask overlay (green) using mask at thumb level for speed ---
    with tiff.TiffFile(mask_path) as tf:
        m = tf.pages[thumb_level].asarray()
    if m.ndim == 3:
        m = m[..., 0]
    binary = (m > 0).astype(np.uint8)

    mask_img = Image.fromarray(binary * 255).resize(thumb.size, Image.NEAREST)

    thumb_rgba = thumb.convert("RGBA")
    overlay = Image.new("RGBA", thumb.size, (0, 255, 0, 0))
    # fast alpha mask: paste green where mask is nonzero
    green = Image.new("RGBA", thumb.size, (0, 255, 0, 80))
    overlay.paste(green, mask=mask_img)
    thumb_rgba = Image.alpha_composite(thumb_rgba, overlay)
    thumb = thumb_rgba.convert("RGB")

    draw = ImageDraw.Draw(thumb)

    # used grid (gray)
    for x0, y0 in used_rois_lvl0:
        x, y = lvl0_to_thumb(x0), lvl0_to_thumb(y0)
        draw.rectangle(
            [x, y, x + roi_size_thumb, y + roi_size_thumb],
            outline=(180, 180, 180),
            width=1,
        )

    # kept tiles (green)
    for x0, y0 in kept_rois_lvl0:
        x, y = lvl0_to_thumb(x0), lvl0_to_thumb(y0)
        draw.rectangle(
            [x, y, x + roi_size_thumb, y + roi_size_thumb], outline=(0, 255, 0), width=2
        )

    # strip anchors (blue)
    r = max(2, roi_size_thumb // 20)
    for x0, y0 in strip_anchors_lvl0:
        cx, cy = lvl0_to_thumb(x0), lvl0_to_thumb(y0)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 128, 255), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(out_path)


# ----------------------------
# Main tiling
# ----------------------------
def tile_wsi(
    wsi_path: Path,
    mask_path: Path,
    output_img_dir: Path,
    output_mask_dir: Path,
    dataset_id: str,
    tile_size: int,
    tile_stride: int,
    target_mag: int,
    labelled_ratio_threshold: float = 0.8,
    min_label_pixels: int | None = None,
    # mask work / speed
    mask_work_level: int | None = None,  # if None: use lowest level
    close_radius: int = 3,  # morphology radius in WORK-level pixels
    fill_holes: bool = True,
    speck_min_area_frac: float = 0.001,  # remove tiny objects as frac of 1 tile area (WORK-level)
    # overview
    save_overview: bool = False,
    overview_dir: Path | None = None,
    overview_max_size: int = 2000,
) -> List[Dict[str, Any]]:
    """
    Fast component-wise biopsy tiling using low-res mask for geometry.

    - Compute grid positions using cleaned binary mask at `mask_work_level` (fast).
    - Discard small components (< 1 tile area at target mag, projected to work-level).
    - For each component, do long-axis snapping + per-strip short-axis snapping.
    - Read image tiles at `target_level` (OpenSlide), using level-0 coordinates.
    - Read semantic mask tiles at `target_level` (tifffile), slice in target-level coords.
    """
    wsi = openslide.OpenSlide(str(wsi_path))

    # --- pick target level based on your existing helper ---
    mpp_x0, mpp_y0 = get_mpp(wsi)
    target_level = select_level_for_magnification(wsi, target_mag)
    ds_target = float(wsi.level_downsamples[target_level])
    wt, ht = wsi.level_dimensions[target_level]

    # --- choose mask work level (lowest by default) ---
    if mask_work_level is None:
        mask_work_level = wsi.level_count - 1
    if not (0 <= mask_work_level < wsi.level_count):
        raise ValueError(f"mask_work_level out of range: {mask_work_level}")

    ds_work = float(wsi.level_downsamples[mask_work_level])
    ww, hw = wsi.level_dimensions[mask_work_level]

    # --- read mask at work level ---
    with tiff.TiffFile(mask_path) as tf:
        mask_work = tf.pages[mask_work_level].asarray()
    if mask_work.ndim == 3:
        mask_work = mask_work[..., 0]
    if mask_work.shape[:2] != (hw, ww):
        raise ValueError(
            f"Mask/WSI mismatch at work level {mask_work_level}: "
            f"WSI={ww}x{hw}, Mask={mask_work.shape[1]}x{mask_work.shape[0]}"
        )

    # --- binary + morphology cleanup ---
    bin_work = mask_work > 0

    if close_radius > 0:
        # disk-ish structure
        y, x = np.ogrid[
            -close_radius : close_radius + 1, -close_radius : close_radius + 1
        ]
        selem = (x * x + y * y) <= close_radius * close_radius
        bin_work = ndi.binary_closing(bin_work, structure=selem)

    if fill_holes:
        bin_work = ndi.binary_fill_holes(bin_work)

    # ROI size projected to work level pixels
    # ROI is defined at target level pixels -> convert to level0 -> convert to work level
    roi_size_lvl0 = int(round(tile_size * ds_target))
    roi_size_work = max(1, int(round(roi_size_lvl0 / ds_work)))
    roi_stride_work = max(1, int(round((tile_stride * ds_target) / ds_work)))

    roi_area_work = roi_size_work * roi_size_work

    # remove tiny specks (before component filtering)
    if speck_min_area_frac > 0:
        min_speck_area = max(1, int(round(speck_min_area_frac * roi_area_work)))
        lab0, n0 = ndi.label(bin_work)
        if n0 > 0:
            sizes = np.bincount(lab0.ravel())
            keep = sizes >= min_speck_area
            keep[0] = False
            bin_work = keep[lab0]

    # connected components on cleaned mask
    labels, ncomp = ndi.label(bin_work)
    if ncomp == 0:
        logger.warning(f"No components after cleanup for {wsi_path.stem}")
        return []

    # drop components smaller than 1 ROI coverage (in WORK-level pixels)
    comp_sizes = np.bincount(labels.ravel())
    min_comp_area = roi_area_work  # "coverage of the tile_size" at target -> projected into work-level
    keep_comp = comp_sizes >= min_comp_area
    keep_comp[0] = False

    kept_ids = np.where(keep_comp)[0]
    logger.info(
        f"{wsi_path.stem}: comps total={ncomp}, kept={len(kept_ids)} "
        f"(min_comp_area_work={min_comp_area}, roi_size_work={roi_size_work})"
    )

    # filtering thresholds for tiles (work-level)
    if min_label_pixels is None:
        # default: 5% of ROI area at TARGET -> convert to WORK approximately
        min_label_pixels_work = int(0.05 * roi_area_work)
    else:
        # user provided min_label_pixels at TARGET pixels; convert to WORK approximately
        min_label_pixels_lvl0 = int(round(min_label_pixels * ds_target * ds_target))
        min_label_pixels_work = max(
            1, int(round(min_label_pixels_lvl0 / (ds_work * ds_work)))
        )

    used_rois_lvl0: List[Tuple[int, int]] = []
    kept_rois_lvl0: List[Tuple[int, int]] = []
    strip_anchors_lvl0: List[Tuple[int, int]] = []

    rows: List[Dict[str, Any]] = []

    def clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(v, hi))

    # helper: map WORK-level (xw,yw) -> level0 coords
    def work_to_lvl0(xw: int, yw: int) -> Tuple[int, int]:
        return int(round(xw * ds_work)), int(round(yw * ds_work))

    # helper: map level0 coords -> TARGET-level coords (for slicing target mask)
    def lvl0_to_target(x0: int, y0: int) -> Tuple[int, int]:
        return int(round(x0 / ds_target)), int(round(y0 / ds_target))

    # read target-level semantic mask once per slide (cheaper than reopening per tile)
    with tiff.TiffFile(mask_path) as tf:
        mask_target = tf.pages[target_level].asarray()
    if mask_target.ndim == 3:
        mask_target = mask_target[..., 0]
    if mask_target.shape[:2] != (ht, wt):
        raise ValueError(
            f"Mask/WSI mismatch at target level {target_level}: "
            f"WSI={wt}x{ht}, Mask={mask_target.shape[1]}x{mask_target.shape[0]}"
        )

    # ---- process each kept component independently ----
    for comp_id in kept_ids.tolist():
        comp = labels == comp_id

        ys, xs = np.nonzero(comp)
        if xs.size == 0:
            continue
        xmin, xmax = int(xs.min()), int(xs.max()) + 1
        ymin, ymax = int(ys.min()), int(ys.max()) + 1
        bw = xmax - xmin
        bh = ymax - ymin
        long_is_x = bw >= bh

        # work in (long, short) coords
        if long_is_x:
            bmin_long, bmax_long, dim_long = xmin, xmax, ww
            bmin_short, bmax_short, dim_short = ymin, ymax, hw
        else:
            bmin_long, bmax_long, dim_long = ymin, ymax, hw
            bmin_short, bmax_short, dim_short = xmin, xmax, ww

        # snap long window to multiple of roi_size_work
        span_long = bmax_long - bmin_long
        n_long = int(np.ceil(span_long / roi_size_work))
        snapped_long = max(roi_size_work, n_long * roi_size_work)
        pad_long = snapped_long - span_long

        start_long = bmin_long - pad_long // 2
        start_long = clamp(start_long, 0, dim_long - snapped_long)
        end_long = start_long + snapped_long

        # iterate strips along long axis
        for p_long in range(start_long, end_long - roi_size_work + 1, roi_stride_work):
            # extract strip of this component
            if long_is_x:
                strip = comp[:, p_long : p_long + roi_size_work]
                short_nonzero = np.where(np.any(strip, axis=1))[0]  # rows (y)
            else:
                strip = comp[p_long : p_long + roi_size_work, :]
                short_nonzero = np.where(np.any(strip, axis=0))[0]  # cols (x)

            if short_nonzero.size == 0:
                continue

            bmin_short_strip = int(short_nonzero.min())
            bmax_short_strip = int(short_nonzero.max()) + 1

            span_short = bmax_short_strip - bmin_short_strip
            n_short = int(np.ceil(span_short / roi_size_work))
            snapped_short = max(roi_size_work, n_short * roi_size_work)
            pad_short = snapped_short - span_short

            start_short = bmin_short_strip - pad_short // 2
            start_short = clamp(start_short, 0, dim_short - snapped_short)
            end_short = start_short + snapped_short

            # record anchor (work -> lvl0)
            if long_is_x:
                xw_anchor, yw_anchor = p_long, start_short
            else:
                xw_anchor, yw_anchor = start_short, p_long
            strip_anchors_lvl0.append(work_to_lvl0(xw_anchor, yw_anchor))

            for p_short in range(
                start_short, end_short - roi_size_work + 1, roi_stride_work
            ):
                if long_is_x:
                    xw, yw = p_long, p_short
                else:
                    xw, yw = p_short, p_long

                # work-level tile mask for filtering (fast)
                tile_comp = comp[yw : yw + roi_size_work, xw : xw + roi_size_work]
                if tile_comp.shape != (roi_size_work, roi_size_work):
                    continue

                label_pixels_w = int(np.count_nonzero(tile_comp))
                ratio_w = label_pixels_w / tile_comp.size

                # keep if ratio ok OR enough absolute label pixels (work-level)
                if (
                    ratio_w < labelled_ratio_threshold
                    and label_pixels_w < min_label_pixels_work
                ):
                    continue

                # map to level0 (for read_region)
                x0, y0 = work_to_lvl0(xw, yw)
                used_rois_lvl0.append((x0, y0))

                # read image at target level (size = roi_size at target)
                tile_img = wsi.read_region(
                    (x0, y0), target_level, (tile_size, tile_size)
                ).convert("RGB")

                # read target-level semantic mask (slice using target coords)
                xt, yt = lvl0_to_target(x0, y0)
                tile_mask_t = mask_target[yt : yt + tile_size, xt : xt + tile_size]
                if tile_mask_t.shape != (tile_size, tile_size):
                    continue

                # optional: also ensure not empty at target (in case of rounding)
                if np.all(tile_mask_t == 0):
                    continue

                img_name = f"{wsi_path.stem}_L{target_level}_{x0}_{y0}.png"
                msk_name = f"{wsi_path.stem}_L{target_level}_{x0}_{y0}.png"

                tile_img.save(output_img_dir / img_name)
                Image.fromarray(tile_mask_t.astype(np.uint8)).save(
                    output_mask_dir / msk_name
                )

                kept_rois_lvl0.append((x0, y0))

                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "sample_id": wsi_path.stem,
                        "group": wsi_path.stem,
                        "image_relpath": str(Path("images") / img_name),
                        "mask_relpath": str(Path("masks_semantic") / msk_name),
                        "width": tile_size,
                        "height": tile_size,
                        "mpp_x": mpp_x0 * ds_target,
                        "mpp_y": mpp_y0 * ds_target,
                        "magnification": target_mag,
                        "target_level": target_level,
                        "target_level_downsample": ds_target,
                        "mask_work_level": mask_work_level,
                        "mask_work_downsample": ds_work,
                        "component_id": int(comp_id),
                    }
                )

    # overview
    if save_overview and overview_dir is not None:
        out_path = overview_dir / f"{wsi_path.stem}_L{target_level}_overview.png"
        _save_overview(
            wsi=wsi,
            mask_path=mask_path,
            used_rois_lvl0=used_rois_lvl0,
            kept_rois_lvl0=kept_rois_lvl0,
            strip_anchors_lvl0=strip_anchors_lvl0,
            roi_size_at_target=tile_size,
            target_level=target_level,
            out_path=out_path,
            overview_max_size=overview_max_size,
        )

    logger.info(
        f"{wsi_path.stem}: tiles kept={len(rows)} "
        f"(work_level={mask_work_level}, target_level={target_level})"
    )
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
@click.option("--tile-size", default=1792, show_default=True)
@click.option("--tile-stride", default=1792, show_default=True)
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
