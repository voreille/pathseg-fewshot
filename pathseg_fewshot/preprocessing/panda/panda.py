from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Increase PIL's decompression bomb limit for large WSI images
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def tile_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    binary_mask: np.ndarray,
    tile_size: int,
    overlap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, float, int, int, int, int]]:
    """
    Tile image, mask, and binary mask into patches.
    
    Args:
        image: Image array (H, W, C)
        mask: Label mask array (H, W)
        binary_mask: Binary tissue mask array (H, W)
        tile_size: Size of tiles (assumes square tiles)
        overlap: Overlap between tiles in pixels
        
    Returns:
        List of (tile_image, tile_mask, coverage, row, col, x, y) tuples where:
        - tile_image: tile from image
        - tile_mask: corresponding tile from label mask
        - coverage: tissue coverage from binary mask
        - row, col: tile position in grid
        - x, y: pixel coordinates of top-left corner
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap
    
    tiles = []
    row = 0
    for y in range(0, h - tile_size + 1, stride):
        col = 0
        for x in range(0, w - tile_size + 1, stride):
            tile_img = image[y:y+tile_size, x:x+tile_size]
            tile_msk = mask[y:y+tile_size, x:x+tile_size]
            tile_binary = binary_mask[y:y+tile_size, x:x+tile_size]
            
            # Calculate coverage from binary mask
            total_pixels = tile_binary.size
            tissue_pixels = np.sum(tile_binary > 0)
            coverage = tissue_pixels / total_pixels if total_pixels > 0 else 0.0
            
            tiles.append((tile_img, tile_msk, coverage, row, col, x, y))
            col += 1
        row += 1
    
    return tiles


def save_tile(
    tile: np.ndarray,
    output_path: Path,
    is_mask: bool = False,
) -> None:
    """Save a tile to disk using OpenCV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if is_mask:
        # Save mask as single-channel PNG
        cv2.imwrite(str(output_path), tile.astype(np.uint8))
    else:
        # OpenCV saves in BGR, so convert RGB to BGR
        tile_bgr = cv2.cvtColor(tile.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), tile_bgr)


@click.command()
@click.option(
    "--csv-path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default="/home/darya/Data/PANDA/data/radboud_train.csv",
    show_default=True,
    help="Path to CSV file with image IDs.",
)
@click.option(
    "--images-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="/home/darya/Data/PANDA/data/train_images",
    show_default=True,
    help="Path to directory containing WSI images.",
)
@click.option(
    "--masks-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="/home/darya/Data/PANDA/data/train_label_masks",
    show_default=True,
    help="Path to directory containing label masks.",
)
@click.option(
    "--binary-masks-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="/home/darya/Data/PANDA/data/binary_mask_radboud",
    show_default=True,
    help="Path to directory containing binary tissue masks.",
)
@click.option(
    "--output-base",
    type=click.Path(file_okay=False, path_type=Path),
    default="/home/darya/Data/PANDA/data",
    show_default=True,
    help="Base output directory for tiles and masks.",
)
@click.option(
    "--tile-size",
    type=int,
    default=512,
    show_default=True,
    help="Size of square tiles to extract.",
)
@click.option(
    "--overlap",
    type=float,
    default=0.0,
    show_default=True,
    help="Overlap between tiles as percentage (0-100). E.g., 50 for 50% overlap.",
)
@click.option(
    "--mpp",
    type=float,
    default=0.5,
    show_default=True,
    help="Microns per pixel (default assumes 20x magnification).",
)
@click.option(
    "--min-coverage",
    type=float,
    default=0.6,
    show_default=True,
    help="Minimum tissue coverage ratio (0-1) to keep a tile. E.g., 0.6 means at least 60% tissue.",
)
def main(
    csv_path: Path,
    images_dir: Path,
    masks_dir: Path,
    binary_masks_dir: Path,
    output_base: Path,
    tile_size: int,
    overlap: float,
    mpp: float,
    min_coverage: float,
) -> None:
    """
    Preprocess and tile the PANDA dataset.
    
    This script will:
    1. Read image IDs from CSV
    2. Tile WSI images and their masks
    3. Use binary masks to calculate tissue coverage
    4. Save tiles that meet minimum coverage threshold
    5. Create unified metadata file
    """
    # Validate parameters
    if not (0 <= overlap <= 100):
        raise ValueError(f"Overlap must be between 0 and 100, got {overlap}")
    if not (0 <= min_coverage <= 1):
        raise ValueError(f"min_coverage must be between 0 and 1, got {min_coverage}")
    
    # Convert overlap percentage to pixels
    overlap_pixels = int(tile_size * overlap / 100.0)
    stride = tile_size - overlap_pixels
    
    click.echo(f"Tiling parameters:")
    click.echo(f"  Tile size: {tile_size}x{tile_size}")
    click.echo(f"  Overlap: {overlap}% ({overlap_pixels} pixels)")
    click.echo(f"  Stride: {stride} pixels")
    click.echo(f"  Min coverage: {min_coverage:.1%} (max {(1-min_coverage):.1%} background)")
    
    if stride <= 0:
        raise ValueError(f"Stride must be > 0. Got stride={stride} (tile_size={tile_size}, overlap={overlap_pixels})")
    
    # Create output directories
    tiles_dir = output_base / "tiles"
    masks_dir_out = output_base / "masks"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    masks_dir_out.mkdir(parents=True, exist_ok=True)
    
    # Read CSV to get image IDs
    click.echo(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    image_ids = df['image_id'].tolist()
    click.echo(f"Found {len(image_ids)} images to process")
    
    # Process each image
    all_metadata_rows = []
    
    # Calculate magnification from mpp (assuming standard relationship)
    # At 20x magnification, mpp is typically 0.5
    magnification = 20.0 * (0.5 / mpp) if mpp > 0 else None
    
    for image_id in tqdm(image_ids, desc="Processing images"):
        # Construct file paths
        img_path = images_dir / f"{image_id}.tiff"
        mask_path = masks_dir / f"{image_id}_mask.tiff"
        binary_mask_path = binary_masks_dir / f"{image_id}.tiff"
        
        # Check if all required files exist
        if not img_path.exists():
            logger.warning(f"Image not found: {image_id}, skipping")
            continue
        if not mask_path.exists():
            logger.warning(f"Mask not found: {image_id}, skipping")
            continue
        if not binary_mask_path.exists():
            logger.warning(f"Binary mask not found: {image_id}, skipping")
            continue
        
        # Load image, mask, and binary mask
        try:
            # Load using PIL for TIFF support
            image = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
            binary_mask = np.array(Image.open(binary_mask_path))
            
            # Ensure RGB format for image
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Ensure mask is single channel
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            # Ensure binary mask is single channel
            if len(binary_mask.shape) == 3:
                binary_mask = binary_mask[:, :, 0]
                
        except Exception as e:
            logger.error(f"Error loading {image_id}: {e}")
            continue
        
        logger.info(f"{image_id}: Image shape {image.shape}, Mask shape {mask.shape}, Binary mask shape {binary_mask.shape}")
        
        # Tile the image and masks
        tiles = tile_image_and_mask(image, mask, binary_mask, tile_size, overlap_pixels)
        
        if len(tiles) == 0:
            logger.warning(f"No tiles extracted for {image_id}")
            continue
        
        # Save tiles and collect metadata
        tiles_kept = 0
        tiles_skipped = 0
        
        for tile_img, tile_msk, coverage, row, col, x, y in tiles:
            tile_name = f"{image_id}_r{row:03d}_c{col:03d}.png"
            
            # Keep tile only if coverage meets threshold
            keep = coverage >= min_coverage
            
            # Debug first few tiles
            if tiles_kept + tiles_skipped < 3:
                logger.info(f"  Tile {tile_name}: coverage: {coverage:.2%}, keep: {keep}")
            
            # Only save tiles and record metadata if they meet coverage threshold
            if keep:
                tile_img_path = tiles_dir / tile_name
                tile_msk_path = masks_dir_out / tile_name
                
                # Save image tile as PNG
                save_tile(tile_img, tile_img_path, is_mask=False)
                
                # Save label mask as PNG
                save_tile(tile_msk, tile_msk_path, is_mask=True)
                
                tiles_kept += 1
                
                # Debug: log first saved tile info
                if tiles_kept == 1:
                    logger.info(f"  First kept tile: {tile_name}, image range [{tile_img.min()}-{tile_img.max()}], mask range [{tile_msk.min()}-{tile_msk.max()}]")
                
                # Add to global metadata (only kept tiles)
                all_metadata_rows.append({
                    'dataset_id': 'panda',
                    'sample_id': tile_name,
                    'group': image_id,
                    'image_relpath': str(Path('tiles') / tile_name),
                    'mask_relpath': str(Path('masks') / tile_name),
                    'width': tile_size,
                    'height': tile_size,
                    'mpp_x': mpp,
                    'mpp_y': mpp,
                    'magnification': magnification if magnification is not None else "",
                })
            else:
                tiles_skipped += 1
        
        click.echo(f"Processed {image_id}: {len(tiles)} tiles ({tiles_kept} kept, {tiles_skipped} skipped - low coverage)")
    
    # Save global metadata
    if all_metadata_rows:
        global_metadata = pd.DataFrame(all_metadata_rows)
        global_metadata_path = output_base / "metadata.csv"
        global_metadata.to_csv(global_metadata_path, index=False)
        click.echo(f"\nSaved metadata to {global_metadata_path}")
        
        # Print summary statistics
        click.echo(f"\n✓ Processing complete!")
        click.echo(f"  Output directory: {output_base}")
        click.echo(f"  Total tiles kept: {len(all_metadata_rows)}")
        click.echo(f"  Unique WSIs: {global_metadata['group'].nunique()}")
    else:
        click.echo(f"\n⚠ No tiles met the coverage threshold!")


if __name__ == "__main__":
    main()
