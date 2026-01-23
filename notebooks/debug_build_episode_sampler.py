import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from pathseg_fewshot.datasets.episode_sampler import (
    EpisodeSpec,
    MinImagesConsumingEpisodeSampler,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_root = Path(os.getenv("DATA_ROOT", "../data/")).resolve()


def main():
    # Load CSV
    df = pd.read_parquet(
        data_root / "index/tile_index_t448_s448/tile_index_t448_s448.parquet"
    )
    AREA_THRESHOLD = 5000  # adjust
    df = df[df["class_area_um2"] >= AREA_THRESHOLD]
    df["class_area_px"] = df["class_area_um2"] / (df["mpp_x"] * df["mpp_y"])

    episode_spec = EpisodeSpec(ways=5, shots=10, queries=2)
    # sampler = ConsumingEpisodeSampler(df, episode_spec, unique_by="row")
    df_filtered = df[~df["is_border_tile"]]
    sampler = MinImagesConsumingEpisodeSampler(
        df_filtered, episode_spec, unique_by="row"
    )
    episode_lists = sampler.build_episode_list(
        episodes_per_dataset=2,
        # dataset_ids=["bcss", "ignite"],
        dataset_ids=["bcss"],
        always_sample_class_id=0,
    )

    episodes_meta = pd.DataFrame(
        columns=["sample_id", "dataset_id", "episode_idx", "class_id", "is_query"]
    )

    episode_lists[0]

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

    episodes_meta.head()


if __name__ == "__main__":
    main()
