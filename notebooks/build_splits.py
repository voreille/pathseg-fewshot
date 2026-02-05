# %%
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
    data_root / "index/tile_index_t448_s448/tile_index_t448_s448.parquet"
)
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
output_dir = fss_data_root / "splits/scenario_3/"
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
def filter_index(input_df, area_threshold=20000, area_threshold_max=30000, min_img_size=448):
    df = input_df.copy()
    df = df[(df["image_width"] >= min_img_size) & (df["image_height"] >= min_img_size)]
    df["class_area_px"] = df["class_area_um2"] / (df["mpp_x"] * df["mpp_y"])
    if "frac_valid" not in df.columns:
        print("Computing frac_valid column")
        df["frac_valid"] = df["valid_pixels"] / (df["w"] * df["h"])

    df = df[df["frac_valid"] >= 0.85]

    df = df[df["class_area_um2"] >= area_threshold]
    df = df[df["class_area_um2"] <= area_threshold_max]
    return df


df_filtered = df[~df["is_border_tile"]]
df_filtered = filter_index(df_filtered, area_threshold=20000, area_threshold_max=30000)
df_filtered.head()

# %%
episode_spec = EpisodeSpec(ways=[6], shots=1, queries=1)


# %%
df_filtered["dataset_dir"].unique()
# %%
sampler = MinImagesConsumingEpisodeSampler(df_filtered, episode_spec, unique_by="row")
episode_lists = sampler.build_episode_list(
    episodes_per_dataset=9,
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
