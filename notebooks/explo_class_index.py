# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
# Load CSV
df = pd.read_parquet(
    "/home/val/workspaces/pathseg-fewshot/data/index/tile_index_t448_s448/tile_index_t448_s448.parquet"
)
# %%
df.head()
# %%
df.info()
# %%

# Filter small areas
AREA_THRESHOLD = 5000  # adjust
df = df[df["class_area_um2"] >= AREA_THRESHOLD]
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
