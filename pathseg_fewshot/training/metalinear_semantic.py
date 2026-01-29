from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR

from pathseg_fewshot.training.lightning_module import LightningModule
from pathseg_fewshot.training.tiler import Tiler


def stack_list(xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(xs, dim=0)


def remap_global_mask_to_episode(
    mask: torch.Tensor,  # [H,W] long-ish
    class_ids: torch.Tensor,  # [N] global ids
    ignore_idx: int,
) -> torch.Tensor:
    """
    Map global ids -> episode indices {0..N-1}. Everything else -> ignore_idx.

    NOTE: If you want "background" as an explicit class, include it in class_ids.
    """
    if mask.dtype != torch.long:
        mask = mask.long()

    # Create a hash-map style remap without Python loops:
    # If your global IDs can be huge (e.g., 1e9), replace with a dict-based remap.
    max_id = int(class_ids.max().item())
    lut = torch.full((max_id + 1,), ignore_idx, dtype=torch.long, device=mask.device)
    lut[class_ids] = torch.arange(
        class_ids.numel(), device=mask.device, dtype=torch.long
    )

    out = mask.clone()
    valid = (out != ignore_idx) & (out >= 0) & (out <= max_id)
    out[valid] = lut[out[valid]]
    out[~valid] = ignore_idx
    return out


class MetaLinearSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,  # put an hist encoder
        num_metrics: int,
        num_classes: int,
        ignore_idx: int,
        img_size: tuple[int, int],
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = False,
        tiler: Optional[Tiler] = None,
    ):
        super().__init__(
            img_size=img_size,
            freeze_encoder=freeze_encoder,
            network=network,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=lr_multiplier_encoder,
            tiler=tiler,
        )

        self.save_hyperparameters(ignore=["network"])

        self.ignore_idx = ignore_idx
        self.poly_lr_decay_power = poly_lr_decay_power

        # self._init_class_weights = weights
        # self.criterion = None

        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)
        patch_size = self.network.patch_size
        self.label_downsampler = nn.AvgPool2d(patch_size, patch_size)

    def _episode_to_tensors(
        self,
        episode: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expects one episode dict (common: DataLoader batch_size=1 with custom collate).
        Returns:
            class_ids: [N]
            support_images: [S,C,H,W]
            support_masks_global: [S,H,W]
            query_images: [Q,C,H,W]
            query_masks_global: [Q,H,W]
        """
        class_ids = episode["class_ids"].to(self.device)

        support_images = (
            stack_list(episode["support_images"]).to(self.device) / 255.0
        )  # TODO: find a better place for this normalization
        query_images = stack_list(episode["query_images"]).to(self.device) / 255.0

        support_masks_global = stack_list(episode["support_masks"]).to(self.device)
        query_masks_global = stack_list(episode["query_masks"]).to(self.device)

        # ensure correct dtypes
        support_masks_global = support_masks_global.long()
        query_masks_global = query_masks_global.long()

        return (
            class_ids,
            support_images,
            support_masks_global,
            query_images,
            query_masks_global,
        )

    def training_step(self, batch, batch_idx):
        loss_total = torch.tensor(0.0, device=self.device)
        for episode in batch:
            (
                class_ids,
                support_images,
                support_masks_global,
                query_images,
                query_masks_global,
            ) = self._episode_to_tensors(episode)

            logits = self.network.forward(
                support_imgs=support_images,
                support_masks=support_masks_global,
                query_imgs=query_images,
                episode_class_ids=class_ids,
            )  # [Q,N,H,W]

            logits = F.interpolate(logits, self.img_size, mode="bilinear")

            # targets = self.to_per_pixel_targets_semantic(query_masks_global, self.ignore_idx)
            # target = torch.stack(query_masks_global).long()

            loss_total += self.criterion(logits, query_masks_global)
        loss_total /= len(batch)
        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total

    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
        is_notebook=False,
    ):
        for episode in batch:
            (
                class_ids,
                support_images,
                support_masks_global,
                query_images,
                query_masks_global,
            ) = self._episode_to_tensors(episode)

            logits = self.network.forward(
                support_imgs=support_images,
                support_masks=support_masks_global,
                query_imgs=query_images,
                episode_class_ids=class_ids,
            )  # [Q,N,H,W]

            logits = F.interpolate(logits, self.img_size, mode="bilinear")

            if is_notebook:
                return logits

            # self.metrics[dataloader_idx].update(logits, query_masks_global)

            if batch_idx == 0:
                name = f"{log_prefix}_{dataloader_idx}_pred_{batch_idx}"
                plot = self.plot_semantic(
                    query_images[0, ...],
                    query_masks_global[0, ...],
                    logits=logits[0],
                )
                # single clean call, no logger assumptions here
                self.log_wandb_image(name, plot, commit=False)

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()

        lr_scheduler = {
            "scheduler": PolynomialLR(
                optimizer,
                int(self.trainer.estimated_stepping_batches),
                self.poly_lr_decay_power,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
