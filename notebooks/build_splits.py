# %%
import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from pathseg_fewshot.datasets.episode import (
    load_episodes_json,
    save_episodes_json,
)
from pathseg_fewshot.datasets.episode_sampler import (
    EpisodeSpec,
    MinImagesConsumingEpisodeSampler,
)

load_dotenv()
data_root = Path(os.getenv("DATA_ROOT", "../data/")).resolve()
fss_data_root = Path(os.getenv("FSS_DATA_ROOT", "../data/fss")).resolve()

# %%
# Load CSV
df = pd.read_parquet(
    data_root / "index/tile_index_t896_s896/tile_index_t896_s896.parquet"
)
# df = pd.read_parquet(
#     data_root / "index/tile_index_t448_s448/tile_index_t448_s448.parquet"
# )
# %%
# Filter small areas
df["class_area_px"] = df["class_area_um2"] / (df["mpp_x"] * df["mpp_y"])
if "frac_valid" not in df.columns:
    print("Computing frac_valid column")
    df["frac_valid"] = df["valid_pixels"] / (df["w"] * df["h"])

df = df[df["frac_valid"] >= 0.75]

AREA_THRESHOLD = 5000  # adjust
df = df[df["class_area_um2"] >= AREA_THRESHOLD]


# %%
g = sns.catplot(
    data=df,
    x="dataset_class_id",
    y="class_area_px",
    col="dataset_id",
    kind="violin",
    inner="quartile",
    scale="width",
    sharey=False,
    col_wrap=2,
    height=4,
    aspect=1.2,
)

g.set_axis_labels("Dataset class ID", "Area (µm²)")
g.set_titles("Dataset: {col_name}")

plt.tight_layout()
plt.show()

# %%
counts = df.groupby(["dataset_id", "dataset_class_id"]).size().reset_index(name="count")
sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=counts,
    x="dataset_class_id",
    y="count",
    col="dataset_id",
    kind="bar",
    col_wrap=2,
    height=4,
    aspect=1.2,
    sharey=False,
)

g.set_axis_labels("Dataset class ID", "Number of occurrences")
g.set_titles("Dataset: {col_name}")

plt.tight_layout()
plt.show()


# %%
output_dir = fss_data_root / "splits/scenario_anorak_2ways_ts896/"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {output_dir}")

# %%
df_split = (
    df.groupby(["sample_id"])
    .aggregate(
        {
            "dataset_id": "first",
            "group": "first",
            "dataset_dir": "first",
            "image_relpath": "first",
            "mask_relpath": "first",
            "mpp_x": "first",
            "mpp_y": "first",
            "image_width": "first",
            "image_height": "first",
        }
    )
    .reset_index()
)

# df_split["split"] = "train"
df_split.loc[df_split["dataset_id"] == "anorak", "split"] = "val"
df_split.loc[df_split["dataset_id"] != "anorak", "split"] = "train"
#

df_split.to_csv(
    output_dir / "split.csv",
    index=False,
)


# %%
df.head()


# %%
def filter_index(
    input_df, area_threshold=0.3, area_threshold_max=0.7, min_img_size=448
):
    df = input_df.copy()
    df = df[(df["image_width"] >= min_img_size) & (df["image_height"] >= min_img_size)]
    df["class_area_px"] = df["class_area_um2"] / (df["mpp_x"] * df["mpp_y"])
    if "frac_valid" not in df.columns:
        print("Computing frac_valid column")
        df["frac_valid"] = df["valid_pixels"] / (df["w"] * df["h"])

    df = df[df["frac_valid"] >= 0.85]

    df = df[df["class_frac_valid"] >= area_threshold]
    df = df[df["class_frac_valid"] <= area_threshold_max]
    return df


df_filtered = df[~df["is_border_tile"]]
df_filtered = filter_index(
    df_filtered, area_threshold=0.25, area_threshold_max=0.75, min_img_size=896
)
df_filtered.head()
df_anorak = df_filtered[df_filtered["dataset_id"] == "anorak"]

# %%
df_anorak["dataset_class_id"].value_counts()
# %%
n_ways = 2
n_classes = df_anorak["dataset_class_id"].nunique()
n_episodes_per_pair = 50
n_shots = 1
n_queries = 1


for pair in itertools.combinations(range(1, n_classes + 1), n_ways):
    episode_list = []
    for i in range(n_episodes_per_pair):
        tiles_support = pd.DataFrame()
        tiles_query = pd.DataFrame()
        for p in pair:
            tiles_support = pd.concat(
                [
                    tiles_support,
                    df_anorak[df_anorak["dataset_class_id"] == p].sample(
                        n=n_shots, replace=False
                    ),
                ]
            )
        used_sample_ids = tiles_support["sample_id"].unique()
        for p in pair:
            for n_q in range(n_queries):
                tiles_query = pd.concat(
                    [
                        tiles_query,
                        df_anorak[
                            (df_anorak["dataset_class_id"] == p)
                            & (~df_anorak["sample_id"].isin(used_sample_ids))
                        ].sample(n=1, replace=False),
                    ]
                )
                used_sample_ids = np.concatenate(
                    [used_sample_ids, tiles_query["sample_id"].unique()]
                )
        episode_list.append((tiles_support, tiles_query))

# %%
episode_spec = EpisodeSpec(ways=[2], shots=1, queries=1)


# %%
df_filtered["dataset_dir"].unique()
# %%
sampler = MinImagesConsumingEpisodeSampler(df_filtered, episode_spec, unique_by="row")
episode_lists = sampler.build_episode_list(
    episodes_per_dataset=10,
    dataset_ids=["anorak"],
)

# %%
episodes_meta = pd.DataFrame(
    columns=["sample_id", "dataset_id", "episode_idx", "class_id", "is_query"]
)

# %%
episode_lists[3]

# %%
for episode_idx, episode in enumerate(episode_lists):
    for s in ["support", "query"]:
        sample_refs = getattr(episode, s)
        for ref in sample_refs:
            episodes_meta = pd.concat(
                [
                    episodes_meta,
                    pd.DataFrame(
                        {
                            "sample_id": [ref.sample_id],
                            "dataset_id": [ref.dataset_id],
                            "episode_idx": [episode_idx],
                            "class_id": [ref.class_id],
                            "is_query": [s == "query"],
                        }
                    ),
                ],
                ignore_index=True,
            )

# %%
episodes_meta.head()
# %%
episodes_meta.loc[episodes_meta["dataset_id"] == "ignite", "sample_id"].nunique()

# %%
episodes_meta.loc[episodes_meta["dataset_id"] == "bcss", "sample_id"].nunique()
# %%
episodes_meta.loc[episodes_meta["dataset_id"] == "anorak", "sample_id"].nunique()

# %%
save_episodes_json(
    output_dir / "val_episodes.json",
    episode_lists,
)

# %%
loaded_episodes = load_episodes_json(output_dir / "val_episodes.json")

# %%
loaded_episodes[0].dataset_dir

# %%
output_dir
# %%
