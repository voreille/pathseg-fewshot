# %%
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


load_dotenv()
data_root = Path(os.getenv("DATA_ROOT", "../data/")).resolve()
fss_data_root = Path(os.getenv("FSS_DATA_ROOT", "../data/fss")).resolve()

# %%
# Load CSV
df = pd.read_parquet(
    "/home/valentin/workspaces/pathseg-fewshot/data/fss/splits/scenario_anorak_2/tile_index_train.parquet"
)

# %%
df.head()
# %%
df = df[df["dataset_id"] != "anorak"]
# %%
df["dataset_id"].unique()
# %%
output_path = "/home/valentin/workspaces/pathseg-fewshot/data/fss/splits/scenario_anorak_2/tile_index_train_wo_anorak.parquet"
df.to_parquet(output_path)

# %%
