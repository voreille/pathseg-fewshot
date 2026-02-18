"""
Create binary masks for PANDA Radboud images.
Binary mask: 1 = tissue, 0 = background
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Increase PIL's decompression bomb limit for large WSI images
Image.MAX_IMAGE_PIXELS = None


def create_binary_mask(image: np.ndarray, threshold: int = 220) -> np.ndarray:
    """
    Create binary mask separating tissue from background.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        threshold: Threshold for background detection (higher = whiter background)
        
    Returns:
        Binary mask as numpy array (H, W) where 1=tissue, 0=background
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Background is typically white/bright in H&E stained slides
    # Tissue regions are darker
    # Create mask where pixels below threshold are tissue (1), others are background (0)
    binary_mask = (gray < threshold).astype(np.uint8)
    
    # Apply morphological operations to clean up the mask
    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Fill small holes
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    return binary_mask


def process_image(image_path: Path, output_path: Path, threshold: int = 220) -> None:
    """
    Load TIFF image, create binary mask, and save it.
    
    Args:
        image_path: Path to input TIFF image
        output_path: Path to output binary mask TIFF
        threshold: Background threshold value
    """
    # Load image using PIL (better for TIFF)
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert BGR to RGB if loaded with 3 channels
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # PIL loads as RGB already, no conversion needed
        pass
    
    # Create binary mask
    binary_mask = create_binary_mask(img_array, threshold=threshold)
    
    # Save binary mask as TIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), binary_mask)


def main():
    # Paths
    csv_path = Path("/home/darya/Data/PANDA/data/radboud_train.csv")
    images_dir = Path("/home/darya/Data/PANDA/data/train_images")
    output_dir = Path("/home/darya/Data/PANDA/data/binary_mask_radboud")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV to get image IDs
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    image_ids = df['image_id'].tolist()
    
    print(f"Found {len(image_ids)} images to process")
    print(f"Output directory: {output_dir}")
    
    # Process each image
    success_count = 0
    failed_count = 0
    
    for image_id in tqdm(image_ids, desc="Creating binary masks"):
        image_path = images_dir / f"{image_id}.tiff"
        output_path = output_dir / f"{image_id}.tiff"
        
        # Skip if already processed
        if output_path.exists():
            success_count += 1
            continue
        
        # Check if image exists
        if not image_path.exists():
            print(f"\nWarning: Image not found: {image_id}")
            failed_count += 1
            continue
        
        # Process image
        try:
            process_image(image_path, output_path, threshold=220)
            success_count += 1
        except Exception as e:
            print(f"\nError processing {image_id}: {e}")
            failed_count += 1
    
    print(f"\n✓ Processing complete!")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
