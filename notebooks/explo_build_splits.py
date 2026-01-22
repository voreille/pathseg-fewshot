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
df[df["dataset_id"] == "anorak"].to_csv(
    data_root / "explorations/anorak_filtered_tiles.csv",
    index=False,
)
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
def sample_support_without_replacement(
    df,
    n_shot,
    random_state=None,
    ignore_class_id={255},
    dataset_id=None,
):
    df = df.copy()
    if dataset_id is not None:
        df = df[df["dataset_id"] == dataset_id]
    else:
        raise ValueError("dataset_id must be specified")

    class_ids = set(df["dataset_class_id"].unique()) - ignore_class_id
    sample_ids = []
    for class_id in class_ids:
        df_class = df[df["dataset_class_id"] == class_id]
        sampled = df_class.sample(n=n_shot, replace=False, random_state=random_state)[
            "sample_id"
        ].tolist()
        sample_ids.extend(sampled)
        df = df[~df["sample_id"].isin(sampled)]
    return sample_ids


# %%
sample_ids = sample_support_without_replacement(
    df,
    n_shot=10,
    random_state=42,
    ignore_class_id={255},
    dataset_id="anorak",
)

# %%
df.columns

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

df_split["support_set"] = np.nan
df_split.loc[df_split["dataset_id"] == "anorak", "split"] = "test"
df_split.loc[df_split["dataset_id"] != "anorak", "split"] = "train"
# %%
df_split[df_split["dataset_id"] == "bcss"].head()

# %%
df_tmp = df.copy()
for support_id in range(3):
    sample_ids = sample_support_without_replacement(
        df_tmp,
        n_shot=10,
        random_state=42 + support_id,
        ignore_class_id={255},
        dataset_id="anorak",
    )

    mask_ids = df_split["sample_id"].isin(sample_ids)
    df_split.loc[mask_ids, "support_set"] = support_id
    df_tmp = df_tmp[~df_tmp["sample_id"].isin(sample_ids)]
# %%
df_split.loc[df_split["support_set"].isna(), "is_query"] = True
df_split.loc[~df_split["support_set"].isna(), "is_query"] = False
df_split.loc[df_split["support_set"].isna(), "support_set"] = -1
df_split.loc[df_split["split"] == "train", "is_query"] = False


# %%
df_split.head()


# %%
output_dir = fss_data_root / "splits/scenario_1/"
output_dir.mkdir(parents=True, exist_ok=True)

df_split.to_csv(
    output_dir / "split.csv",
    index=False,
)

# %%

episode_spec = EpisodeSpec(ways=5, shots=10, queries=2)
sampler = MinImagesConsumingEpisodeSampler(df, episode_spec, unique_by="row")
episode_lists = sampler.build_episode_list(
    episodes_per_dataset=2,
    dataset_ids=["bcss", "ignite"],
    always_sample_class_id=0,  # ensure background is present
)

# %%
episodes_meta = pd.DataFrame(
    columns=["sample_id", "dataset_id", "episode_idx", "class_id", "is_query"]
)

# %%
episode_lists[0]

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
bcss_sample_ids = (
    episodes_meta.loc[episodes_meta["dataset_id"] == "bcss", "sample_id"]
    .unique()
    .tolist()
)
# %%
ignite_sample_ids = (
    episodes_meta.loc[episodes_meta["dataset_id"] == "ignite", "sample_id"]
    .unique()
    .tolist()
)


# %%
df_split.loc[
    df_split["sample_id"].isin(bcss_sample_ids) & (df_split["dataset_id"] == "bcss"),
    "split",
] = "val"

# %%
df_split.loc[
    df_split["sample_id"].isin(ignite_sample_ids)
    & (df_split["dataset_id"] == "ignite"),
    "split",
] = "val"
# %%
df_split.to_csv(
    output_dir / "split.csv",
    index=False,
)

# %%
save_episodes_json(
    output_dir / "val_episodes.json",
    episode_lists,
)

# %%
loaded_episodes = load_episodes_json(output_dir / "val_episodes.json")

# %%
loaded_episodes[0].support[0]

# %%
