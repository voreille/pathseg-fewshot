# %%
import os
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from pathseg_fewshot.datasets.episode import EpisodeSpec
from pathseg_fewshot.datasets.episode_datasets import OnTheFlyEpisodeDataset
from pathseg_fewshot.datasets.episode_sampler import StatelessEpisodeSampler
from pathseg_fewshot.datasets.transforms import CustomTransforms

load_dotenv()

# %%
dataset_root = Path(os.getenv("FSS_DATA_ROOT", "../data/fss")).resolve()
class_index = pd.read_parquet(
    "/home/valentin/workspaces/pathseg-fewshot/data/index/tile_index_t672_s448/tile_index_t672_s448.parquet"
)


# %%
def train_collate(batch):
    imgs, targets = [], []

    for img, target in batch:
        imgs.append(img)
        targets.append(target)

    return torch.stack(imgs), targets


dataloader_kwargs = {
    "persistent_workers": False,
    "num_workers": 4,
    "pin_memory": True,
    "batch_size": 1,
    "prefetch_factor": None,
}


# %%
episode_spec = EpisodeSpec(ways=[5], shots=10, queries=15)
episode_sampler = StatelessEpisodeSampler(class_index=class_index, spec=episode_spec)
dataset = OnTheFlyEpisodeDataset(
    root_dir=dataset_root,
    sampler=episode_sampler,
    episodes_per_epoch=1000,
    transform=CustomTransforms(img_size=(448, 448), scale_range=(0.8, 1.2)),
)
dataloader = DataLoader(dataset, collate_fn=train_collate, **dataloader_kwargs)

# %%
episode = dataset[0]
# %%
