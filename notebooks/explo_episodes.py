# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dotenv import load_dotenv

from pathseg_fewshot.datasets.fss_data_module import FSSDataModule
from pathseg_fewshot.tools.save_episodes_as_tiles import save_episodes_as_tiles

load_dotenv()
data_root = Path(os.getenv("DATA_ROOT", "../data/")).resolve()
fss_data_root = Path(os.getenv("FSS_DATA_ROOT", "../data/fss")).resolve()
workdir = Path("../").resolve()

# %%
data_module = FSSDataModule(
    root=fss_data_root,
    tile_index_parquet=data_root
    / "index/tile_index_t672_s448/tile_index_t672_s448.parquet",
    split_csv=workdir / "data/fss/splits/scenario_1/split.csv",
    val_episodes_json=workdir / "data/fss/splits/scenario_1/val_episodes.json",
    ways=[1],
    shots=5,
    queries=1,
    img_size=(448, 448),
    batch_size=1,
    num_workers=4,
    episodes_per_epoch=1000,
)
# %%
data_module.setup("fit")
# %%
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# %%
episodes = []
n_show = 5  # change
for i, batch in enumerate(val_loader):
    episodes.append(batch)
    if i + 1 >= n_show:
        break

# %%
# --- helpers for visualization ---
def _to_numpy_image(img: torch.Tensor) -> np.ndarray:
    """
    img: [3,H,W] uint8 or float
    -> [H,W,3] float in [0,1]
    """
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu()
    else:
        x = torch.as_tensor(img)

    # tv_tensors.Image is still a Tensor subclass; this works.
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        # if already 0..255 floats, normalize; else assume 0..1
        if x.max() > 1.5:
            x = x / 255.0

    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).contiguous()
    return x.numpy()

def _to_numpy_mask(mask: torch.Tensor) -> np.ndarray:
    """
    mask: [H,W] long
    """
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask)
    return m

def _unwrap_batch(batch):
    """
    Handles common batch structures when batch_size=1:
    - batch is [ {episode_dict} ]
    - batch is { ... } but some fields are length-1 lists/tensors
    """
    # Case 1: DataLoader returned a list with a single dict
    if isinstance(batch, list):
        if len(batch) != 1:
            raise ValueError(f"Expected batch list of length 1, got {len(batch)}")
        batch = batch[0]

    if not isinstance(batch, dict):
        raise TypeError(f"Expected dict-like batch, got {type(batch)}")

    out = dict(batch)

    # Unwrap common fields if they are batched
    if isinstance(out.get("dataset_id"), (list, tuple)) and len(out["dataset_id"]) == 1:
        out["dataset_id"] = out["dataset_id"][0]

    if isinstance(out.get("seed"), torch.Tensor) and out["seed"].numel() == 1:
        out["seed"] = int(out["seed"].item())
    elif isinstance(out.get("seed"), (list, tuple)) and len(out["seed"]) == 1:
        s = out["seed"][0]
        out["seed"] = int(s.item()) if isinstance(s, torch.Tensor) else int(s)

    if isinstance(out.get("class_ids"), torch.Tensor) and out["class_ids"].ndim == 2 and out["class_ids"].shape[0] == 1:
        out["class_ids"] = out["class_ids"][0]
    elif isinstance(out.get("class_ids"), (list, tuple)) and len(out["class_ids"]) == 1:
        out["class_ids"] = out["class_ids"][0]

    # Sometimes images/masks get an extra batch dimension too:
    for k in ["support_images", "support_masks", "query_images", "query_masks"]:
        if isinstance(out.get(k), list) and len(out[k]) == 1 and isinstance(out[k][0], list):
            out[k] = out[k][0]

    return out


def plot_episode(
    episode_batch: dict,
    *,
    ignore_idx: int = 255,
    alpha: float = 0.45,
    max_items: int | None = None,
):
    ep = _unwrap_batch(episode_batch)

    class_ids = ep["class_ids"]
    if isinstance(class_ids, torch.Tensor):
        class_ids_list = class_ids.detach().cpu().tolist()
    else:
        class_ids_list = list(class_ids)

    support_imgs = ep["support_images"]
    support_masks = ep["support_masks"]
    query_imgs = ep["query_images"]
    query_masks = ep["query_masks"]

    ns = len(support_imgs)
    nq = len(query_imgs)

    if max_items is not None:
        ns = min(ns, max_items)
        nq = min(nq, max_items)

    # 3 columns: image, overlay, mask
    rows = ns + nq
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    title = f"dataset_id={ep.get('dataset_id')} | seed={ep.get('seed')} | class_ids(global)={class_ids_list} | remap: bg=0, classes=1..K, ignore={ignore_idx}"
    fig.suptitle(title, y=0.995, fontsize=12)

    def _plot_row(r, img_t, mask_t, label):
        img = _to_numpy_image(img_t)
        mask = _to_numpy_mask(mask_t)

        # image
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{label}: image")
        axes[r, 0].axis("off")

        # overlay (hide ignore by turning it into NaN)
        overlay = mask.astype(np.float32)
        overlay[overlay == ignore_idx] = np.nan
        axes[r, 1].imshow(img)
        axes[r, 1].imshow(overlay, alpha=alpha)
        axes[r, 1].set_title(f"{label}: overlay (alpha={alpha})")
        axes[r, 1].axis("off")

        # mask only
        axes[r, 2].imshow(overlay)  # ignore hidden
        axes[r, 2].set_title(f"{label}: mask (ignore hidden)")
        axes[r, 2].axis("off")

    r = 0
    for i in range(ns):
        _plot_row(r, support_imgs[i], support_masks[i], f"SUPPORT {i}")
        r += 1
    for i in range(nq):
        _plot_row(r, query_imgs[i], query_masks[i], f"QUERY {i}")
        r += 1

    plt.tight_layout()
    plt.show()

# %%
# --- visualize the collected episodes ---
for i, batch in enumerate(episodes):
    print(f"Episode {i}")
    plot_episode(batch, ignore_idx=255, alpha=0.45)

# %%
# --- OPTIONAL: export episodes as tiles (if your helper expects an iterable of episode dicts) ---
# out_dir = workdir / "debug/episode_tiles"
# out_dir.mkdir(parents=True, exist_ok=True)
# save_episodes_as_tiles(episodes, out_dir=out_dir)  # adjust signature if needed
# print(f"Saved to: {out_dir}")

# %%
episodes[2][0]["support_masks"][1].unique()
# %%
