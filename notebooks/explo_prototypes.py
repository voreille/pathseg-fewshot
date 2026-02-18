# %%
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from sklearn.manifold import TSNE

from pathseg_fewshot.datasets.fss_data_module import FSSDataModule
from pathseg_fewshot.models.few_shot_segmenter import FewShotSegmenter
from pathseg_fewshot.training.metalinear_semantic import MetaLinearSemantic

load_dotenv()

data_root = Path(os.getenv("DATA_ROOT", "../data/")).resolve()
fss_data_root = Path(os.getenv("FSS_DATA_ROOT", "../data/fss")).resolve()
workdir = Path("../").resolve()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# %%
data_module = FSSDataModule(
    root=fss_data_root,
    tile_index_parquet="/home/valentin/workspaces/pathseg-fewshot/data/fss/splits/scenario_anorak_2/tile_index_train_wo_anorak.parquet",
    split_csv="/home/valentin/workspaces/pathseg-fewshot/data/fss/splits/scenario_anorak_2/split.csv",
    val_episodes_json="/home/valentin/workspaces/pathseg-fewshot/data/fss/splits/scenario_anorak_2/test_episodes.json",
    ways=[2],
    shots=1,
    queries=1,
    img_size=(896, 896),
    batch_size=1,
    num_workers=4,
    episodes_per_epoch=1000,
)
# %%
data_module.setup("fit")
# %%
train_loader = data_module.train_dataloader()
val_loaders = data_module.val_dataloader()
val_loader = val_loaders[0]

# %%
episodes = []
n_show = 5  # change
for i, batch in enumerate(val_loader):
    episodes.append(batch)
    if i + 1 >= n_show:
        break

# %%
run_path = Path(
    "/home/valentin/workspaces/pathseg-fewshot/runs/fewshot-experiment/version_3"
)
config = yaml.safe_load((run_path / "config.yaml").read_text())
checkpoint_path = "/home/valentin/workspaces/pathseg-fewshot/runs/fewshot-experiment/version_3/checkpoints/20260212-173643/epoch=0-step=2500.ckpt"
network = FewShotSegmenter.from_config(config["model"]["init_args"]["network"])
pl_module = MetaLinearSemantic.load_from_checkpoint(checkpoint_path, network=network)
network = pl_module.network
# %%
network.to(device)
network.eval()
# %%


def stack_list(xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(xs, dim=0)


def episode_to_tensors(
    episode: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Expects one episode dict (common: DataLoader batch_size=1 with custom collate).
    Returns:
        class_ids: [N]
        support_images: [S,C,H,W]
        support_masks_global: [S,H,W]
        query_images: [Q,C,H,W]
        query_masks_global: [Q,H,W]
    """
    class_ids = episode["class_ids"].to(device)

    support_images = stack_list(episode["support_images"]).to(device) / 255.0
    query_images = stack_list(episode["query_images"]).to(device) / 255.0

    support_masks_global = stack_list(episode["support_masks"]).to(device)
    query_masks_global = stack_list(episode["query_masks"]).to(device)

    # ensure correct dtypes
    support_masks_global = support_masks_global.long()
    query_masks_global = query_masks_global.long()

    return {
        "class_ids": class_ids,
        "support_images": support_images,
        "support_masks_global": support_masks_global,
        "query_images": query_images,
        "query_masks_global": query_masks_global,
    }


# %%
episode = episode_to_tensors(episodes[0][0], device)

# %%
episode["class_ids"]

# %%
ctx = network.fit_support(
    support_imgs=episode["support_images"],
    support_masks=episode["support_masks_global"],
    episode_class_ids=episode["class_ids"],
)

# %%
support_features = network.encode(episode["support_images"])
support_features.shape


# %%
support_labels, support_valids = network.encode_support_labels(episode["support_masks_global"], num_fg_classes=2)
# %%
prototypes_bank = network.meta.bank
prototypes_bank.shape

# %%

# %%
def to_tokens_features(F: torch.Tensor) -> torch.Tensor:
    # [S,D,Ht,Wt] -> [S,T,D] or passthrough if already [S,T,D]
    if F.ndim == 4:
        S, D, Ht, Wt = F.shape
        return F.permute(0, 2, 3, 1).reshape(S, Ht * Wt, D)
    if F.ndim == 3:
        return F
    raise ValueError(F"Unexpected support_features shape: {F.shape}")

def to_tokens_probs(Y: torch.Tensor) -> torch.Tensor:
    # [S,C,Ht,Wt] -> [S,T,C] or passthrough if already [S,T,C]
    if Y.ndim == 4:
        S, C, Ht, Wt = Y.shape
        return Y.permute(0, 2, 3, 1).reshape(S, Ht * Wt, C)
    if Y.ndim == 3:
        return Y
    raise ValueError(F"Unexpected support_labels shape: {Y.shape}")

def to_tokens_valid(V: torch.Tensor) -> torch.Tensor:
    # [S,Ht,Wt] -> [S,T] or passthrough if already [S,T]
    if V.ndim == 3:
        S, Ht, Wt = V.shape
        return V.reshape(S, Ht * Wt)
    if V.ndim == 2:
        return V
    raise ValueError(F"Unexpected support_valids shape: {V.shape}")

@torch.no_grad()
def tsne_support_tokens_vs_bank(
    support_features: torch.Tensor,
    support_probs: torch.Tensor,
    support_valids: torch.Tensor,
    bank: torch.Tensor,
    class_ids: torch.Tensor | None = None,   # episode class IDs (optional; for nicer labels)
    thr: float = 0.9,
    max_tokens: int = 5000,
    seed: int = 0,
    perplexity: int = 30,
):
    # --- flatten to token space ---
    F = to_tokens_features(support_features)      # [S,T,D]
    P = to_tokens_probs(support_probs)            # [S,T,C]
    V = to_tokens_valid(support_valids).bool()    # [S,T]

    S, T, D = F.shape
    C = P.shape[-1]

    # --- valid tokens only ---
    F = F[V]   # [N,D]
    P = P[V]   # [N,C]

    # --- hard label by threshold ---
    conf, lab = P.max(dim=-1)                    # [N], [N]
    lab = torch.where(conf >= thr, lab, torch.full_like(lab, -1))  # -1 = unconfident

    # --- subsample tokens for t-SNE ---
    N = F.shape[0]
    g = torch.Generator(device=F.device).manual_seed(seed)
    if N > max_tokens:
        idx = torch.randperm(N, generator=g, device=F.device)[:max_tokens]
        F = F[idx]
        lab = lab[idx]
        conf = conf[idx]

    # --- bank vectors (ensure 2D [N_bank, D]) ---
    bank2 = bank
    if bank2.ndim == 3:
        # if your bank is [N_bank, K, D], flatten K into N_bank*K
        bank2 = bank2.reshape(-1, bank2.shape[-1])
    assert bank2.ndim == 2 and bank2.shape[-1] == D, (bank2.shape, D)

    # --- stack for t-SNE ---
    X = torch.cat([F, bank2.to(F.device)], dim=0).float().cpu().numpy()
    token_n = F.shape[0]
    bank_n = bank2.shape[0]

    # --- t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, (X.shape[0] - 1) // 3)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    Z = tsne.fit_transform(X)

    Z_tok = Z[:token_n]
    Z_bank = Z[token_n:]

    lab_np = lab.cpu().numpy()

    # --- plotting ---
    plt.figure(figsize=(9, 7))

    # tokens: plot per label (including -1)
    uniq = np.unique(lab_np)
    for u in uniq:
        m = lab_np == u
        name = "unconfident" if u == -1 else f"class {u}"
        if class_ids is not None and u != -1:
            # map token label index -> episode class id (optional cosmetic)
            try:
                name = f"class_idx {u} (id={int(class_ids[u])})"
            except Exception:
                pass
        plt.scatter(Z_tok[m, 0], Z_tok[m, 1], s=6, alpha=0.5, label=name)

    # bank: stars
    plt.scatter(Z_bank[:, 0], Z_bank[:, 1], s=80, marker="*", label=f"bank ({bank_n})")

    plt.title(f"t-SNE: support tokens (thr={thr}, n={token_n}) + bank prototypes (n={bank_n})")
    plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

# token features used by the head
features = to_tokens_features(support_features)            # [S,T,D]
features = features.reshape(-1, features.shape[-1])

# apply same centering as head (mu computed in fit_support)
mu = ctx.get("center", None)
if mu is not None:
    features = features - mu

# apply same normalization as head
features = torch.nn.functional.normalize(features, dim=-1)
bank_prototypes = torch.nn.functional.normalize(network.meta.bank, dim=-1)
# ---- call it on your episode ----
tsne_support_tokens_vs_bank(
    support_features=support_features,
    support_probs=support_labels,      # your soft labels in token space
    support_valids=support_valids,     # bool valid (label != 255)
    bank=bank_prototypes,
    class_ids=episode["class_ids"],
    thr=0.9,
    max_tokens=6000,
    seed=0,
    perplexity=30,
)


# %%
support_labels.shape
# %%



@torch.no_grad()
def coverage_stats(tokens: torch.Tensor, bank: torch.Tensor):
    # tokens: [N,E], bank: [K,E] assumed already centered if you use center
    tokens = F.normalize(tokens, dim=-1)
    bank = F.normalize(bank, dim=-1)

    sim = tokens @ bank.t()               # [N,K]
    best = sim.max(dim=1).values          # [N]

    return {
        "mean": best.mean().item(),
        "p10": best.kthvalue(int(0.10 * best.numel())).values.item(),
        "p50": best.median().item(),
        "p90": best.kthvalue(int(0.90 * best.numel())).values.item(),
    }

# use SAME tokens you used in tsne
X = features  # [N,E] already centered+normalized in your snippet
stats = coverage_stats(X, bank_prototypes)
print(stats)


# %%
K, E = bank_prototypes.shape
Prand = F.normalize(torch.randn(K, E, device=bank_prototypes.device), dim=-1)
print("rand:", coverage_stats(X, Prand))
print("learned:", coverage_stats(X, bank_prototypes))

# %%
@torch.no_grad()
def bank_diversity(bank: torch.Tensor):
    B = F.normalize(bank, dim=-1)
    G = B @ B.t()
    K = G.shape[0]
    off = G[~torch.eye(K, dtype=torch.bool, device=G.device)]
    return {
        "mean_abs_offdiag": off.abs().mean().item(),
        "max_offdiag": off.max().item(),
        "min_offdiag": off.min().item(),
    }

print(bank_diversity(prototypes_bank))

# %%
