# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathseg_fewshot.datasets.episode_sampler import EpisodeSampler, EpisodeSpec

# %%
# Load CSV
df = pd.read_csv("/home/valentin/data/fss/class_index_preview.csv")

# Filter small areas
AREA_THRESHOLD = 5000  # adjust
df = df[df["area_um2"] >= AREA_THRESHOLD]
df["area_px"] = df["area_um2"] / (df["mpp"] ** 2)

# %%
df.dataset_id.unique()

# %%
sampler = EpisodeSampler(df, seed=42)
# %%
spec = EpisodeSpec(ways=5, shots=10, queries=2, crop_size=448, seed=123)
# %%
episode = sampler.sample_episode(spec)
# %%
print("Support set:")
for ref in episode.support:
    print(f"  Dataset: {ref.dataset_id}, Class: {ref.class_id}, Sample: {ref.sample_id}")   
# %%

query_ids = [ref.sample_id for ref in episode.query]
support_ids = [ref.sample_id for ref in episode.support]

# %%
overlap = set(query_ids) & set(support_ids)
print(f"Overlap between support and query sets: {overlap}")  # Should be empty


# %%
query_class_ids = set([ref.class_id for ref in episode.query])
support_class_ids = set([ref.class_id for ref in episode.support])
print(f"Query class IDs: {query_class_ids}")
print(f"Support class IDs: {support_class_ids}")

# %%
