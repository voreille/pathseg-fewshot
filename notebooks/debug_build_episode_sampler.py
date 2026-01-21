import pandas as pd

from pathseg_fewshot.datasets.episode_sampler import (
    ConsumingEpisodeSampler,
    MinImagesConsumingEpisodeSampler,
    EpisodeSpec,
)


def main():
    # Load CSV
    df = pd.read_parquet(
        "/home/valentin/workspaces/pathseg-fewshot/data/index/tile_index_t448_s448/tile_index_t448_s448.parquet"
    )
    AREA_THRESHOLD = 5000  # adjust
    df = df[df["class_area_um2"] >= AREA_THRESHOLD]
    df["class_area_px"] = df["class_area_um2"] / (df["mpp_x"] * df["mpp_y"])

    episode_spec = EpisodeSpec(ways=5, shots=10, queries=2, crop_size=448)
    # sampler = ConsumingEpisodeSampler(df, episode_spec, unique_by="row")
    sampler = MinImagesConsumingEpisodeSampler(df, episode_spec, unique_by="row")
    episode_lists = sampler.build_episode_list(
        episodes_per_dataset=2, dataset_ids=["bcss", "ignite"]
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
