import torch
from torch import nn
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F


class Transforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        scale_range: tuple[float, float],
        max_brightness_delta=32,
        max_contrast_factor=0.5,
        saturation_factor=0.5,
        max_hue_delta=18,
    ):
        super().__init__()

        self.img_size = img_size

        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()

        self.scale_range = scale_range

        self.random_crop = T.RandomCrop(img_size)

    def random_factor(self, factor, center=1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, brightness_factor=self.random_factor(self.max_brightness_factor)
            )

        return img

    def contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(
                img,
                contrast_factor=self.random_factor(self.max_contrast_factor),
            )

        return img

    def saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img,
                saturation_factor=self.random_factor(self.max_saturation_factor),
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(
                img,
                hue_factor=self.random_factor(self.max_hue_delta, center=0.0),
            )

        return img

    def color_jitter(self, img):
        img = self.brightness(img)

        if torch.rand(()) < 0.5:
            img = self.contrast(img)
            img = self.saturation_and_hue(img)
        else:
            img = self.saturation_and_hue(img)
            img = self.contrast(img)

        return img

    def pad(self, img, target: dict):
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding, padding_mode="edge")
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def crop(self, img, target: dict):
        img_crop, target_crop = self.random_crop(img, target)

        mask_sums = target_crop["masks"].sum(dim=[-2, -1])
        non_empty_mask = mask_sums > 0

        if non_empty_mask.sum() == 0:
            return self.crop(img, target)

        target_crop["masks"] = target_crop["masks"][non_empty_mask]
        target_crop["labels"] = target_crop["labels"][non_empty_mask]

        return img_crop, target_crop

    def forward(self, img, target: dict):
        img = self.color_jitter(img)

        c, h, w = img.shape
        # this change is to control the scale for varying input size with same magnification
        # similar results can be obtained by swapping the h and w, but find it less intuitive
        target_size = max(h, w)
        img, target = T.ScaleJitter(
            target_size=(target_size, target_size),
            # target_size=(w, h),
            scale_range=self.scale_range,
            antialias=True,
        )(img, target)

        img, target = self.random_horizontal_flip(img, target)

        img, target = self.pad(img, target)

        return self.crop(img, target)


class SemanticTransforms(nn.Module):
    """
    Transforms for semantic segmentation with per-pixel labels.

    Expected input:
      img: Tensor[C,H,W] (uint8 or float)
      target: {"mask": Tensor[H,W] (long/int), ...}
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        scale_range: tuple[float, float],
        *,
        ignore_idx: int = 255,
        background_idx: int
        | None = 0,  # set to None if you don't want to treat bg specially
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
        min_valid_fraction: float = 0.001,  # minimum fraction of valid pixels in a crop
        max_crop_tries: int = 20,
    ):
        super().__init__()

        self.img_size = img_size
        self.scale_range = scale_range

        self.ignore_idx = int(ignore_idx)
        self.background_idx = background_idx

        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()
        self.random_vertical_flip = T.RandomVerticalFlip()

        # RandomCrop in v2 can take (img, target) if target contains tv_tensors / tensors
        self.random_crop = T.RandomCrop(img_size)

        self.min_valid_fraction = float(min_valid_fraction)
        self.max_crop_tries = int(max_crop_tries)

    def random_factor(self, factor, center=1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    # ---- Color jitter: image only ----
    def brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, brightness_factor=self.random_factor(self.max_brightness_factor)
            )
        return img

    def contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(
                img,
                contrast_factor=self.random_factor(self.max_contrast_factor),
            )
        return img

    def saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img,
                saturation_factor=self.random_factor(self.max_saturation_factor),
            )
        if torch.rand(()) < 0.5:
            img = F.adjust_hue(
                img,
                hue_factor=self.random_factor(self.max_hue_delta, center=0.0),
            )
        return img

    def color_jitter(self, img):
        img = self.brightness(img)
        if torch.rand(()) < 0.5:
            img = self.contrast(img)
            img = self.saturation_and_hue(img)
        else:
            img = self.saturation_and_hue(img)
            img = self.contrast(img)
        return img

    # ---- Geom helpers: apply to img + semantic mask ----
    def pad(self, img, target: dict):
        mask = target["mask"]

        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]  # left, top, right, bottom

        img = F.pad(img, padding, padding_mode="edge")

        # For semantic masks, pad with ignore index (NOT edge replication)
        mask = F.pad(mask, padding, fill=self.ignore_idx)
        target["mask"] = mask

        return img, target

    def _crop_is_ok(self, mask_crop: torch.Tensor) -> bool:
        # mask_crop: [H,W]
        valid = mask_crop != self.ignore_idx
        valid_frac = valid.float().mean().item()
        if valid_frac < self.min_valid_fraction:
            return False

        # Optional: ensure not purely background (if background_idx provided)
        if self.background_idx is not None:
            # pixels that are valid and not background
            fg = valid & (mask_crop != self.background_idx)
            if fg.sum().item() == 0:
                return False

        return True

    def crop(self, img, target: dict):
        # Try a few times to avoid empty crops
        for _ in range(self.max_crop_tries):
            img_crop, target_crop = self.random_crop(img, target)
            if self._crop_is_ok(target_crop["mask"]):
                return img_crop, target_crop

        # If we fail, just return last crop (or you could fall back to center crop)
        return img_crop, target_crop

    def forward(self, img, target: dict):
        """
        target must contain:
          - "mask": Tensor[H,W] (long/int)
        """
        # color jitter on image only
        img = self.color_jitter(img)

        c, h, w = img.shape
        target_size = max(h, w)

        # ScaleJitter will resize image and mask together (mask gets nearest interpolation)
        img, target = T.ScaleJitter(
            target_size=(target_size, target_size),
            scale_range=self.scale_range,
            antialias=True,
        )(img, target)

        img, target = self.pad(img, target)

        img, target = self.random_horizontal_flip(img, target)
        img, target = self.random_vertical_flip(img, target)

        return self.crop(img, target)
