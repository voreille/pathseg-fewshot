# %%
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import tifffile as tiff


# %%
def load_label_mask(mask_path: Path) -> np.ndarray:
    """
    Load a semantic segmentation mask as integer class IDs.

    Important:
    - Do NOT convert 'P' to 'L' (palette -> grayscale destroys IDs).
    - Fail fast on visualization-style masks (values near 255).
    """
    with Image.open(mask_path) as im:
        if im.mode == "P":
            mask = np.array(im, dtype=np.int32)  # indices = class ids
        elif im.mode == "L":
            mask = np.array(im, dtype=np.int32)
        else:
            raise ValueError(
                f"Unsupported mask mode {im.mode} for {mask_path}. "
                "Expected 'P' or 'L'. If RGB/RGBA, need a color->id mapping."
            )

    return mask


# %%
panda_path = Path("/mnt/nas7/data/Personal/Valentin/PANDA/data")
# %%
df = pd.read_csv(panda_path / "train.csv")

# %%
df.head()
# %%
imgage_ids = df["image_id"].tolist()
# %%

image_id = imgage_ids[0]
image_path = panda_path / "train_images" / f"{image_id}.tiff"

image = openslide.OpenSlide(str(image_path))

# %%
image.level_downsamples
# %%
tile = image.read_region((0, 0), 2, (512, 512))  # RGBA
# %%
tile_np = np.array(tile)
tile_np.shape

# %%
mask_path = panda_path / "train_label_masks" / f"{image_id}_mask.tiff"
mask = openslide.OpenSlide(str(mask_path))

# %%
str(mask_path)
# %%
mpp_x = image.properties.get(openslide.PROPERTY_NAME_MPP_X)
mpp_y = image.properties.get(openslide.PROPERTY_NAME_MPP_Y)

if mpp_x is not None:
    mpp_x = float(mpp_x)
if mpp_y is not None:
    mpp_y = float(mpp_y)

print("MPP X:", mpp_x)
print("MPP Y:", mpp_y)
# %%
mpp_x = mask.properties.get(openslide.PROPERTY_NAME_MPP_X)
mpp_y = mask.properties.get(openslide.PROPERTY_NAME_MPP_Y)

if mpp_x is not None:
    mpp_x = float(mpp_x)
if mpp_y is not None:
    mpp_y = float(mpp_y)

print("MPP X:", mpp_x)
print("MPP Y:", mpp_y)

# %%
img_thumbnail = image.get_thumbnail((512, 512))
mask_thumbnail = mask.get_thumbnail((512, 512))
mask_np = np.array(mask_thumbnail)
mask_np = mask_np[:, :, 0]  # if RGB with same values, take one channel
mask_np[mask_np > 6] = 1  # crude check for visualization-style masks

# %%
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_thumbnail)
plt.title("Image Thumbnail")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(mask_np)
plt.title("Mask Thumbnail")
plt.axis("off")
plt.tight_layout()
plt.show()
# %%
np.unique(mask_np)

# %%
str(mask_path)

# %%

s = openslide.OpenSlide(mask_path)
print("vendor:", s.properties.get("openslide.vendor"))
print("level count:", s.level_count)
print("level dims:", s.level_dimensions)
print("downsamples:", s.level_downsamples)

# tile sizes if present
for k, v in s.properties.items():
    if "tile" in k.lower() or "tiff" in k.lower():
        print(k, "=", v)
# %%
s = openslide.OpenSlide(mask_path)
level = 0
W, H = s.level_dimensions[level]

im0 = np.array(s.read_region((0, 0), level, (W, H)))  # RGBA
im0 = im0[..., 0]  # if mask is grayscale stored in R (common). Otherwise adjust.

thumb = im0
# %%
plt.imshow(thumb)
# %%


def read_mask_level(mask_path: Path, level: int = 0):
    with tiff.TiffFile(mask_path) as tf:
        return tf.pages[level].asarray()[:, :, 0]


mask0 = read_mask_level(mask_path, 0)
mask2 = read_mask_level(mask_path, 2)
# %%
plt.imshow(mask0)

# %%
plt.imshow(mask2)
# %%
np.unique(mask2)

