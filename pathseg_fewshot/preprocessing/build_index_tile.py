# build_tile_index.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def _read_metadata(dataset_dir: Path) -> pd.DataFrame:
    p_parquet = dataset_dir / "metadata.parquet"
    p_csv = dataset_dir / "metadata.csv"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Missing metadata.(parquet|csv) in {dataset_dir}")


def _iter_anchors_1d(length: int, tile: int, stride: int) -> List[int]:
    """Base anchors + border anchor (length-tile)."""
    if length < tile:
        # either skip sample or treat as one tile anchored at 0 (your choice)
        return [0]
    anchors = list(range(0, length - tile + 1, stride))
    border = length - tile
    if anchors[-1] != border:
        anchors.append(border)
    return anchors


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L")
        return np.array(im, dtype=np.uint8)


@click.command()
@click.option(
    "--root-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--include",
    type=str,
    default=None,
    help="Comma-separated dataset folder names to include.",
)
@click.option(
    "--exclude",
    type=str,
    default=None,
    help="Comma-separated dataset folder names to exclude.",
)
@click.option(
    "--tile-size", type=int, required=True, help="Tile size in pixels (square)."
)
@click.option(
    "--stride", type=int, default=None, help="Stride in pixels (default: tile-size)."
)
@click.option(
    "--min-class-pixels",
    type=int,
    default=500,
    show_default=True,
    help="Keep (tile,class) rows only if class_pixels >= this.",
)
@click.option(
    "--ignore-id",
    type=int,
    default=255,
    show_default=True,
    help="Ignore label id (optional).",
)
@click.option(
    "--write-per-dataset/--no-write-per-dataset", default=True, show_default=True
)
@click.option("--concat-global/--no-concat-global", default=False, show_default=True)
def main(
    root_dir: Path,
    output_dir: Path,
    include: Optional[str],
    exclude: Optional[str],
    tile_size: int,
    stride: Optional[int],
    min_class_pixels: int,
    ignore_id: int,
    write_per_dataset: bool,
    concat_global: bool,
) -> None:
    """
    Build a tile-level index for fast few-shot episode sampling.

    Expects per-dataset prepared outputs:
      dataset/
        metadata.(parquet|csv)
        images/...
        masks_semantic/...
    """
    stride = int(stride) if stride is not None else int(tile_size)

    output_dir = output_dir / f"tile_index_t{tile_size}_s{stride}"
    output_dir.mkdir(parents=True, exist_ok=True)

    include_set = set(x.strip() for x in include.split(",")) if include else None
    exclude_set = set(x.strip() for x in exclude.split(",")) if exclude else set()

    dataset_dirs = []
    for p in sorted(root_dir.iterdir()):
        if not p.is_dir():
            continue
        if include_set is not None and p.name not in include_set:
            continue
        if p.name in exclude_set:
            continue
        if not ((p / "metadata.parquet").exists() or (p / "metadata.csv").exists()):
            continue
        dataset_dirs.append(p)

    if not dataset_dirs:
        raise click.ClickException("No datasets found.")

    all_tiles: List[pd.DataFrame] = []

    for ds_dir in dataset_dirs:
        out_ds_dir = output_dir / ds_dir.name
        out_ds_dir.mkdir(parents=True, exist_ok=True)
        dataset_id = ds_dir.name
        meta = _read_metadata(ds_dir).copy()
        if "dataset_id" not in meta.columns:
            meta["dataset_id"] = dataset_id

        rows = []
        for _, r in tqdm(meta.iterrows(), total=len(meta), desc=f"{dataset_id}"):
            sample_id = str(r["sample_id"])
            group = str(r.get("group", sample_id))

            mask_rel = Path(str(r["mask_relpath"]))
            mask_path = ds_dir / mask_rel
            mask = _load_mask(mask_path)
            H, W = mask.shape
            if H != r["height"] or W != r["width"]:
                raise ValueError(
                    f"Size mismatch for {dataset_id}/{sample_id}: "
                    f"metadata ({r['width']}x{r['height']}) vs mask ({W}x{H})"
                )

            xs = _iter_anchors_1d(W, tile_size, stride)
            ys = _iter_anchors_1d(H, tile_size, stride)

            # optional: precompute which classes exist in the whole mask to skip extra work
            present = set(np.unique(mask).tolist())
            if ignore_id in present:
                present.remove(ignore_id)
            if 0 in present:
                present.remove(0)

            for y in ys:
                for x in xs:
                    tile = mask[y : y + tile_size, x : x + tile_size]

                    # ignore filtering (optional)
                    if ignore_id is not None:
                        valid = tile != ignore_id
                        valid_pixels = int(valid.sum())
                        if valid_pixels == 0:
                            continue
                    else:
                        valid_pixels = tile_size * tile_size

                    # Count pixels per label in the tile
                    vals, counts = np.unique(tile, return_counts=True)
                    count_map = {int(v): int(c) for v, c in zip(vals, counts)}

                    # Emit one row per class_id (excluding background/ignore)
                    for cid in present:
                        class_pixels = count_map.get(int(cid), 0)
                        if class_pixels < min_class_pixels:
                            continue

                        rows.append(
                            {
                                "dataset_id": dataset_id,
                                "sample_id": sample_id,
                                "group": group,
                                "image_relpath": str(r["image_relpath"]),
                                "mask_relpath": str(r["mask_relpath"]),
                                "dataset_dir": str(ds_dir),
                                "mpp_x": float(r["mpp_x"]),
                                "mpp_y": float(r["mpp_y"]),
                                "magnification": int(r["magnification"]),
                                "image_width": int(W),
                                "image_height": int(H),
                                "tile_size": int(tile_size),
                                "stride": int(stride),
                                "x": int(x),
                                "y": int(y),
                                "w": int(tile_size),
                                "h": int(tile_size),
                                "tile_id": f"{x}_{y}",
                                "dataset_class_id": int(cid),
                                "class_pixels": int(class_pixels),
                                "class_area_um2": float(class_pixels)
                                * (r["mpp_x"] * r["mpp_y"]),
                                "valid_pixels": int(valid_pixels),
                                "class_frac_valid": float(
                                    class_pixels / max(1, valid_pixels)
                                ),
                                "is_border_tile": bool(
                                    x == W - tile_size or y == H - tile_size
                                ),
                            }
                        )

        tile_df = pd.DataFrame(rows)
        if write_per_dataset:
            tile_df.to_parquet(
                out_ds_dir / f"tile_index_t{tile_size}_s{stride}.parquet", index=False
            )

        all_tiles.append(tile_df)

    if concat_global:
        global_df = pd.concat(all_tiles, ignore_index=True)
        global_df.to_parquet(
            output_dir / f"tile_index_t{tile_size}_s{stride}.parquet", index=False
        )

    click.echo("Done.")


if __name__ == "__main__":
    main()
