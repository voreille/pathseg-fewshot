# %%
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from pathseg_fewshot.datasets.episode import (
    EpisodeRefs,
    load_episodes_json,
    save_episodes_json,
)
from pathseg_fewshot.datasets.episode_sampler import (
    EpisodeSpec,
    MinImagesConsumingEpisodeSampler,
)

# %%
# Load CSV
df = pd.read_parquet(
    "/home/valentin/workspaces/pathseg-fewshot/data/index/tile_index_t448_s448/tile_index_t448_s448.parquet"
)
# %%
# Filter small areas
# AREA_THRESHOLD = 5000  # adjust
# df = df[df["class_area_um2"] >= AREA_THRESHOLD]
df["class_area_px"] = df["class_area_um2"] / (df["mpp_x"] * df["mpp_y"])


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
output_dir = Path("/home/valentin/workspaces/pathseg-fewshot/data/splits/scenario_1/")
output_dir.mkdir(parents=True, exist_ok=True)

df_split.to_csv(
    output_dir / "split.csv",
    index=False,
)

# %%

episode_spec = EpisodeSpec(ways=5, shots=10, queries=2)
sampler = MinImagesConsumingEpisodeSampler(df, episode_spec, unique_by="row")
episode_lists = sampler.build_episode_list(
    episodes_per_dataset=2, dataset_ids=["bcss", "ignite"]
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
def save_episodes_as_tiles(
    root_dir: Path, output_dir: Path, episodes: List[EpisodeRefs]
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = []
    for e_idx, e in enumerate(episodes):
        episode_dir = output_dir / f"episode_{e_idx:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        support_dir = episode_dir / "support"
        support_dir.mkdir(parents=True, exist_ok=True)

        query_dir = episode_dir / "query"
        query_dir.mkdir(parents=True, exist_ok=True)

        for r in e.support:

            img_path = episode_dir / f"support_{r.sample_id}_image.png"
            mask_path = episode_dir / f"support_{r.sample_id}_mask.png"

            img = Image.open(root_dir / r.dataset_id / r.image_relpath)
            mask = Image.open(root_dir / r.dataset_id / r.mask_relpath)

            img.save(img_path)
            mask.save(mask_path)
        for r in e.query:
            d["query"].append(
                {
                    "dataset_id": r.dataset_id,
                    "sample_id": r.sample_id,
                    "image_relpath": r.image_relpath,
                    "mask_relpath": r.mask_relpath,
                    "class_id": int(r.class_id),
                    "crop": r.crop,
                }
            )
        payload.append(d)

    with (output_dir / "episodes.json").open("w") as f:
        json.dump(payload, f, indent=2)
