# TODO: maybe put the construction of the network inside the constructor so it is easier to save/load and stored in parameters
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR

from pathseg_fewshot.training.lightning_module import LightningModule
from pathseg_fewshot.training.losses import CEDiceLoss
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
        num_classes: int,
        ignore_idx: int,
        img_size: tuple[int, int],
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = False,
        tiler: Optional[Tiler] = None,
        loss: str = "ce_dice",
        add_loss_par: bool = False,
        loss_bank_div_coeff: float = 1e-3,
        loss_bank_usage_coeff: float = 1e-3,
        loss_sparse_coeff: float = 1e-3,
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
        self.num_classes = num_classes
        self.add_loss_par = add_loss_par
        self.loss_bank_div_coeff = loss_bank_div_coeff
        self.loss_bank_usage_coeff = loss_bank_usage_coeff
        self.loss_sparse_coeff = loss_sparse_coeff

        self.save_hyperparameters(ignore=["network"])

        self.ignore_idx = ignore_idx
        self.poly_lr_decay_power = poly_lr_decay_power

        # self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)
        if loss == "ce_dice":
            self.criterion = CEDiceLoss(ignore_index=self.ignore_idx)
        elif loss == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)
        else:
            raise ValueError(f"Unsupported loss type: {loss}")
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
        # batch is a list of episodes (your current assumption)
        device = self.device
        loss_total = torch.tensor(0.0, device=device)
        loss_bank_div = torch.tensor(0.0, device=device)
        loss_bank_usage = torch.tensor(0.0, device=device)

        # --- 1) collect all images for one encoder forward
        all_imgs = []
        episode_specs = []  # to slice back later

        for episode in batch:
            (
                class_ids,
                support_imgs,
                support_masks,
                query_imgs,
                query_masks,
            ) = self._episode_to_tensors(episode)

            S = support_imgs.shape[0]
            Q = query_imgs.shape[0]

            start = len(all_imgs)
            all_imgs.extend([support_imgs, query_imgs])

            episode_specs.append(
                dict(
                    class_ids=class_ids,  # [K]
                    support_masks=support_masks,  # [S,H,W]
                    query_masks=query_masks,  # [Q,H,W]
                    S=S,
                    Q=Q,
                    start=start,  # start index in "all_imgs blocks"
                )
            )

        imgs_cat = torch.cat(all_imgs, dim=0)  # because each element is already a batch
        feats_cat = self.network.encode(imgs_cat)  # [B,T,Fp]  (FewShotSegmenter.encode)

        # Now slice per episode: because we concatenated support then query as blocks,
        # we need to track feature offsets in "image count" space, not list indices.
        # Easier: rebuild offsets by accumulating image counts:
        offset = 0
        for spec in episode_specs:
            S, Q = spec["S"], spec["Q"]
            support_feats = feats_cat[offset : offset + S]  # [S,T,Fp]
            query_feats = feats_cat[offset + S : offset + S + Q]  # [Q,T,Fp]
            offset += S + Q

            class_ids = spec["class_ids"]
            support_masks = spec["support_masks"]  # [S,H,W]
            query_masks = spec["query_masks"]  # [Q,H,W]

            # --- 3) build token-level support labels & valid mask (on token grid)
            K = int(class_ids.numel())  # fg ways
            support_labels_tok, support_valid_tok = self.network.encode_support_labels(
                support_masks=support_masks,
                num_fg_classes=K,
            )  # [S,T,K+1], [S,T]

            # --- 4) run episode head on features
            # logits_flat = self.network.forward_features(
            #     support_features=support_feats,  # [S,T,Fp]
            #     support_labels=support_labels_tok,  # [S,T,C]
            #     query_features=query_feats,  # [Q,T,Fp]
            #     support_valid=support_valid_tok,  # [S,T]
            # )  # [Q*T, C] where C=K+1
            ctx = self.network.fit_support_features(
                support_features=support_feats,
                support_labels=support_labels_tok,
                support_valid=support_valid_tok,
            )
            logits_flat = self.network.predict_query_features(
                query_features=query_feats,
                support_ctx=ctx,
            )

            # reshape to token grid [Q,C,Hp,Wp]
            logits = self.network.unflatten_logits(logits_flat, num_query=Q)
            logits = F.interpolate(logits, self.img_size, mode="bilinear")
            pred_masks = logits.argmax(dim=1)  # [Q,H,W]

            loss_total += self.criterion(logits, query_masks)

            if self.add_loss_par:
                pred_labels_tok, pred_valid_tok = self.network.encode_support_labels(
                    support_masks=pred_masks,
                    num_fg_classes=K,
                )  # [S,T,K+1], [S,T]
                logits_par = self.network.forward_features(
                    support_features=query_feats,  # [S,T,Fp]
                    support_labels=pred_labels_tok,  # [S,T,C]
                    query_features=support_feats,  # [Q,T,Fp]
                    support_valid=pred_valid_tok,  # [S,T]
                )  # [Q*T, C] where C=K+1
                # CE expects [Q,C,Hp,Wp] vs [Q,Hp,Wp]
                logits_par = self.network.unflatten_logits(logits_par, num_query=S)
                logits_par = F.interpolate(logits_par, self.img_size, mode="bilinear")
                loss_total += self.criterion(logits_par, support_masks)

            zero = loss_total.new_tensor(0.0)
            loss_bank_div += ctx.get("aux_bank_div", zero)
            loss_bank_usage += ctx.get("aux_bank_usage", zero)
            loss_total += self.loss_sparse_coeff * ctx.get("aux_sparse", zero)

        loss_total = loss_total / len(batch)
        loss_bank_div = loss_bank_div / len(batch)
        loss_bank_usage = loss_bank_usage / len(batch)

        loss_total += self.loss_bank_div_coeff * loss_bank_div
        loss_total += self.loss_bank_usage_coeff * loss_bank_usage

        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)
        self.log("train_loss_bank_div", loss_bank_div, sync_dist=True, prog_bar=False)
        self.log("train_loss_bank_usage", loss_bank_usage, sync_dist=True, prog_bar=False)
        return loss_total

    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
        is_notebook=False,
    ):
        logits_list = []
        query_masks_list = []
        for episode in batch:
            (
                class_ids,
                support_images,
                support_masks_global,
                query_images,
                query_masks_global,
            ) = self._episode_to_tensors(episode)

            Q = query_images.shape[0]
            logits = self.network.forward(
                support_imgs=support_images,
                support_masks=support_masks_global,
                query_imgs=query_images,
                episode_class_ids=class_ids,
            )  # [Q,N,H,W]

            logits = F.interpolate(logits, self.img_size, mode="bilinear")
            logits_list.extend(logits[i, ...] for i in range(Q))
            query_masks_list.extend([query_masks_global[i, ...] for i in range(Q)])

            if is_notebook:
                return logits

            if batch_idx == 0:
                name = f"{log_prefix}_{dataloader_idx}_pred_{batch_idx}"
                plot = self.plot_semantic(
                    query_images[0, ...],
                    query_masks_global[0, ...],
                    logits=logits[0],
                )
                # single clean call, no logger assumptions here
                # self.log_wandb_image(name, plot, commit=False)
                self.log_tb_image(name, plot)

        self.update_metrics(logits_list, query_masks_list, dataloader_idx)

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    # def configure_optimizers(self):
    #     optimizer = super().configure_optimizers()

    #     lr_scheduler = {
    #         "scheduler": PolynomialLR(
    #             optimizer,
    #             int(self.trainer.estimated_stepping_batches),
    #             self.poly_lr_decay_power,
    #         ),
    #         "interval": "step",
    #     }

    #     return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_optimizers(self):
        # --- collect encoder params (if present)
        optimizer = super().configure_optimizers()
        # enc_params = (
        #     list(self.network.encoder.parameters())
        #     if hasattr(self.network, "encoder")
        #     else []
        # )
        # enc_ids = {id(p) for p in enc_params}

        # # --- split params into decay / no_decay for base + encoder
        # base_decay, base_no_decay = [], []
        # enc_decay, enc_no_decay = [], []

        # for name, p in self.named_parameters():
        #     if not p.requires_grad:
        #         continue

        #     is_encoder = id(p) in enc_ids
        #     is_no_decay = (
        #         name.endswith(".bias")
        #         or "norm" in name.lower()
        #         or "layernorm" in name.lower()
        #     )

        #     if is_encoder:
        #         (enc_no_decay if is_no_decay else enc_decay).append(p)
        #     else:
        #         (base_no_decay if is_no_decay else base_decay).append(p)

        # optimizer = AdamW(
        #     [
        #         {
        #             "params": base_decay,
        #             "lr": self.lr,
        #             "weight_decay": self.weight_decay,
        #         },
        #         {"params": base_no_decay, "lr": self.lr, "weight_decay": 0.0},
        #         {
        #             "params": enc_decay,
        #             "lr": self.lr * self.lr_multiplier_encoder,
        #             "weight_decay": self.weight_decay,
        #         },
        #         {
        #             "params": enc_no_decay,
        #             "lr": self.lr * self.lr_multiplier_encoder,
        #             "weight_decay": 0.0,
        #         },
        #     ],
        #     betas=(0.9, 0.95),  # good default for transformer-ish heads
        # )

        lr_scheduler = {
            "scheduler": PolynomialLR(
                optimizer,
                int(self.trainer.estimated_stepping_batches),
                self.poly_lr_decay_power,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_validation_start(self):
        dm = self.trainer.datamodule

        if hasattr(dm, "val_pairs"):
            num_metrics = len(dm.val_pairs)
        else:
            num_metrics = 1

        self.init_metrics_semantic(
            num_classes=self.hparams.num_classes,
            ignore_idx=self.ignore_idx,
            num_metrics=num_metrics,
        )

        self.metrics = self.metrics.to(self.device)

    def _on_eval_epoch_end_semantic(self, log_prefix):
        for metric_idx, metric in enumerate(self.metrics):
            iou_per_class = metric.compute()
            metric.reset()

            # pair label from datamodule
            pair = None
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "val_pairs"):
                pair = dm.val_pairs[metric_idx]  # e.g. (2,5)

            tag = f"{pair[0]}_{pair[1]}" if pair is not None else str(metric_idx)

            for iou_idx, iou in enumerate(iou_per_class):
                self.log(f"{log_prefix}_pair_{tag}_iou_{iou_idx}", iou, sync_dist=True)

            self.log(
                f"{log_prefix}_pair_{tag}_miou",
                float(iou_per_class.mean()),
                sync_dist=True,
            )

    @torch.no_grad()
    def collect_tokens_for_bank_init(
        self,
        dataloader,
        *,
        max_tokens: int = 50_000,
        max_episodes: int = 200,
        include_query: bool = True,
        seed: int = 0,
    ) -> torch.Tensor:
        """
        Collect a pool of token features [N, E] from multiple episodes to init the meta bank.
        Uses the current network encoder (`self.network.encode`).

        Notes:
          - Respects your episode structure (batch = list of episodes).
          - Uses your `_episode_to_tensors` (so same normalization etc.)
          - Subsamples to `max_tokens`.
        """
        self.network.eval()

        chunks = []
        n_total = 0

        # We use a CPU buffer list for safety, then move to device at the end
        for i, batch in enumerate(dataloader):
            if i >= max_episodes:
                break

            # batch is a list of episodes
            for episode in batch:
                class_ids, support_imgs, support_masks, query_imgs, query_masks = (
                    self._episode_to_tensors(episode)
                )

                if include_query:
                    imgs = torch.cat([support_imgs, query_imgs], dim=0)  # [S+Q,C,H,W]
                else:
                    imgs = support_imgs

                feats = self.network.encode(imgs)  # [B,T,E]
                feats = feats.reshape(-1, feats.shape[-1])  # [N,E]

                chunks.append(feats.detach().cpu())
                n_total += feats.shape[0]

                if n_total >= max_tokens:
                    break

            if n_total >= max_tokens:
                break

        if len(chunks) == 0:
            raise RuntimeError("collect_tokens_for_bank_init: collected 0 tokens.")

        X = torch.cat(chunks, dim=0)  # [N,E] on CPU
        if X.shape[0] > max_tokens:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(X.shape[0], generator=g)[:max_tokens]
            X = X[idx]
        
        self.network.train()
        return X.to(self.device)

    def on_fit_start_old(self):
        dm = self.trainer.datamodule

        if hasattr(dm, "val_pairs"):
            num_metrics = len(dm.val_pairs)
        else:
            num_metrics = 1

        self.init_metrics_semantic(
            num_classes=self.hparams.num_classes,
            ignore_idx=self.ignore_idx,
            num_metrics=num_metrics,
        )

    def on_fit_start(self):
        # --- (1) optionally init bank BEFORE metrics init (either order is fine)
        meta = getattr(self.network, "meta", None)
        has_bank = meta is not None and hasattr(meta, "bank")

        # Detect distributed
        is_dist = dist.is_available() and dist.is_initialized()
        rank0 = (not is_dist) or dist.get_rank() == 0

        # Avoid re-init when resuming from checkpoint
        # (global_step > 0 is the simplest robust guard)
        should_init = has_bank and (self.global_step == 0)

        if should_init:
            if rank0:
                train_loader = self.trainer.datamodule.train_dataloader()

                X = self.collect_tokens_for_bank_init(
                    train_loader,
                    max_tokens=50_000,
                    max_episodes=200,
                    include_query=True,
                    seed=0,
                )

                # This method must exist on your meta head:
                # meta.init_bank_from_tokens(tokens, ...)
                meta.init_bank_from_tokens(
                    X,
                    max_tokens=50_000,
                    kmeans_iters=25,
                    seed=0,
                )

            # Broadcast the initialized bank to all ranks (DDP-safe)
            if is_dist:
                dist.broadcast(meta.bank.data, src=0)

        # --- (2) keep your original metrics init logic
        dm = self.trainer.datamodule
        if hasattr(dm, "val_pairs"):
            num_metrics = len(dm.val_pairs)
        else:
            num_metrics = 1

        self.init_metrics_semantic(
            num_classes=self.hparams.num_classes,
            ignore_idx=self.ignore_idx,
            num_metrics=num_metrics,
        )

    def on_before_optimizer_step(self, optimizer):
        meta = getattr(self.network, "meta", None)
        if meta is not None and hasattr(meta, "bank"):
            with torch.no_grad():
                meta.bank.data = F.normalize(meta.bank.data, dim=-1)
