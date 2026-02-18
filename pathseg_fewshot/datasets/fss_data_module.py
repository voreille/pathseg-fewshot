from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from lightning.pytorch.utilities import rank_zero_info
from torch import nn
from torch.utils.data import DataLoader

from pathseg_fewshot.datasets.episode import EpisodeSpec, load_episodes_json
from pathseg_fewshot.datasets.episode_datasets import (
    EpisodeListDataset,
    OnTheFlyEpisodeDataset,
)
from pathseg_fewshot.datasets.episode_sampler import StatelessEpisodeSampler
from pathseg_fewshot.datasets.lightning_data_module import LightningDataModule
from pathseg_fewshot.datasets.transforms import (
    SemanticTransforms,
)


class FSSDataModule(LightningDataModule):
    def __init__(
        self,
        root,
        tile_index_parquet: str | Path,
        split_csv: str | Path,
        val_episodes_json: Optional[str | Path] = None,
        ways: Optional[list[int]] = None,  # default to [1] later
        shots: int = 1,
        queries: int = 1,
        num_workers: int = 0,
        img_size: tuple[int, int] = (448, 448),
        batch_size: int = 1,
        num_metrics: int = 1,
        scale_range=(0.8, 1.2),
        ignore_idx: int = 255,
        prefetch_factor: int = 2,
        transform: Optional[nn.Module] = None,
        episodes_per_epoch: int = 1000,
        min_valid_frac: float = 0.85,
        min_class_ratio: float = 0.25,
        max_class_ratio: float = 0.75,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            num_metrics=num_metrics,
            ignore_idx=ignore_idx,
            img_size=img_size,
            prefetch_factor=prefetch_factor,
        )

        self.split_df = pd.read_csv(split_csv)
        self.val_episodes_list = []
        if val_episodes_json is not None:
            self.val_episodes_list = load_episodes_json(val_episodes_json)

        self._check_split_df()

        self.root_dir = Path(root).resolve()

        self.min_valid_frac = min_valid_frac
        self.min_class_ratio = min_class_ratio
        self.max_class_ratio = max_class_ratio

        self.tile_index = pd.read_parquet(tile_index_parquet)
        self.tile_index = self._filter_index(self.tile_index)

        if ways is None:
            ways = [1]
        else:
            ways = ways

        self.spec = EpisodeSpec(ways=ways, shots=shots, queries=queries)
        self.episodes_per_epoch = episodes_per_epoch

        rank_zero_info(
            f"[FSSDataModule] Initializing datamodule with split_csv = {split_csv}"
        )
        rank_zero_info(
            f"[FSSDataModule] Initializing datamodule with batch_size = {batch_size}"
        )

        self.save_hyperparameters(ignore=["transforms"])
        if transform is not None:
            self.transform = transform
        else:
            self.transform = SemanticTransforms(
                img_size=img_size,
                scale_range=scale_range,
            )

    def _filter_index(self, df):
        df["class_area_px"] = df["class_area_um2"] / (df["mpp_x"] * df["mpp_y"])
        if "frac_valid" not in df.columns:
            print("Computing frac_valid column")
            df["frac_valid"] = df["valid_pixels"] / (df["w"] * df["h"])

        df = df[df["frac_valid"] >= self.min_valid_frac]

        df = df[df["class_frac_valid"] >= self.min_class_ratio]
        df = df[df["class_frac_valid"] <= self.max_class_ratio]
        return df

    def _check_split_df(self) -> None:
        required_cols = {"sample_id", "dataset_id", "split"}
        missing_cols = required_cols - set(self.split_df.columns)
        if missing_cols:
            raise ValueError(f"split_df is missing required columns: {missing_cols}")

        if len(self.val_episodes_list) > 0:
            val_sample_ids = set()
            for episode in self.val_episodes_list:
                val_sample_ids.update(set(episode.get_sample_ids()))

            df_val_ids = set(
                self.split_df[self.split_df["split"] == "val"]["sample_id"].tolist()
            )

            if len(df_val_ids) == 0:
                rank_zero_info(
                    "[FSSDataModule] Warning: "
                    "split_df has no samples for 'val' "
                    "split setting the val_ids from val_episodes_json"
                )
                self.split_df.loc[
                    self.split_df["sample_id"].isin(val_sample_ids), "split"
                ] = "val"

            elif not val_sample_ids.issubset(df_val_ids):
                raise ValueError(
                    f"Mismatch between val sample_ids in split_df and val_episodes_json. "
                    f"split_df has {len(df_val_ids)} ids, val_episodes_json has {len(val_sample_ids)} ids."
                )

    def _get_split_ids(self):
        train_ids = self.split_df[self.split_df["split"] == "train"][
            "sample_id"
        ].tolist()
        val_ids = self.split_df[self.split_df["split"] == "val"]["sample_id"].tolist()
        test_ids = self.split_df[self.split_df["split"] == "test"]["sample_id"].tolist()
        return train_ids, val_ids, test_ids

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        train_ids, _, _ = self._get_split_ids()

        self.tile_index = self.tile_index[self.tile_index["sample_id"].isin(train_ids)]

        if stage in ("fit", "validate", None):
            train_sampler = StatelessEpisodeSampler(
                class_index=self.tile_index,
                spec=self.spec,
            )
            self.train_dataset = OnTheFlyEpisodeDataset(
                self.root_dir,
                sampler=train_sampler,
                episodes_per_epoch=self.episodes_per_epoch,
                transform=self.transform,
            )
            episodes_by_pair = defaultdict(list)
            for ep in self.val_episodes_list:
                key = tuple(sorted(ep.class_ids))  # e.g. (2, 5)
                episodes_by_pair[key].append(ep)

            self.val_pairs = sorted(episodes_by_pair.keys())
            self.val_datasets = [
                EpisodeListDataset(self.root_dir, episodes=episodes_by_pair[pair])
                for pair in self.val_pairs
            ]
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda x: x,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return [
            DataLoader(ds, collate_fn=lambda x: x, **self.dataloader_kwargs)
            for ds in self.val_datasets
        ]

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset,
    #         collate_fn=lambda x: x,
    #         **self.dataloader_kwargs,
    #     )

    def predict_dataloader(self):
        # make sure datasets exist
        if not hasattr(self, "val_dataset") or not hasattr(self, "test_dataset"):
            self.setup(stage="predict")

        loaders, splits = [], []

        if getattr(self, "val_dataset", None) is not None:
            loaders.append(self.val_dataloader())
            splits.append("val")

        if getattr(self, "test_dataset", None) is not None:
            loaders.append(self.test_dataloader())
            splits.append("test")

        # store the mapping in the datamodule
        self.predict_splits = splits
        return loaders
