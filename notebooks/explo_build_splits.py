# %%
import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import seaborn as sns

# %%
# Load CSV
df = pd.read_csv("/home/valentin/data/fss/class_index_preview.csv")

# Filter small areas
AREA_THRESHOLD = 5000  # adjust
df = df[df["area_um2"] >= AREA_THRESHOLD]
df["area_px"] = df["area_um2"] / (df["mpp"] ** 2)


# %%
g = sns.catplot(
    data=df,
    x="dataset_class_id",
    y="area_px",
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
df_split = (
    df.groupby(["sample_id"])
    .aggregate(
        {
            "dataset_id": "first",
            "group": "first",
            "dataset_dir": "first",
            "image_relpath": "first",
            "mask_relpath": "first",
            "mpp": "first",
            "width": "first",
            "height": "first",
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
df_split.to_csv(
    "/home/valentin/workspaces/pathseg-fewshot/data/splits/scenario_1/split.csv", index=False
)

# %%
