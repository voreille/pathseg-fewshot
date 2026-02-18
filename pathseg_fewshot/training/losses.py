import torch
import torch.nn.functional as F
from torch import nn


def soft_dice_loss(
    logits: torch.Tensor,  # [B,C,H,W]
    target: torch.Tensor,  # [B,H,W] long
    ignore_index: int,
    eps: float = 1e-6,
    include_background: bool = True,
) -> torch.Tensor:
    # mask out ignored pixels
    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.new_tensor(0.0)

    B, C, H, W = logits.shape

    probs = logits.softmax(dim=1)  # [B,C,H,W]

    # Replace ignored target values with 0 so one_hot doesn't crash
    target_clean = target.clone()
    target_clean[~valid] = 0

    # one-hot: [B,H,W,C] -> [B,C,H,W]
    target_1h = F.one_hot(target_clean, num_classes=C).permute(0, 3, 1, 2).float()

    # apply valid mask to both probs and target
    valid_f = valid.unsqueeze(1).float()  # [B,1,H,W]
    probs = probs * valid_f
    target_1h = target_1h * valid_f

    # sum over spatial dims
    dims = (0, 2, 3)  # batch+spatial (macro over classes)
    inter = (probs * target_1h).sum(dims)  # [C]
    denom = probs.sum(dims) + target_1h.sum(dims)  # [C]
    dice_per_class = (2.0 * inter + eps) / (denom + eps)  # [C]

    if not include_background:
        dice_per_class = dice_per_class[1:]  # drop class 0

    return 1.0 - dice_per_class.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        dice = soft_dice_loss(logits, target, self.ignore_index)
        return ce + self.dice_weight * dice
