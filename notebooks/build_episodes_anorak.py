import itertools
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from pathseg_fewshot.datasets.episode import EpisodeRefs, SampleRef, save_episodes_json


def filter_index(
    input_df: pd.DataFrame,
    area_threshold: float = 0.3,
    area_threshold_max: float = 0.7,
    min_img_size: int = 448,
    frac_valid_threshold: float = 0.85,
) -> pd.DataFrame:
    df = input_df.copy()

    df = df[(df["image_width"] >= min_img_size) & (df["image_height"] >= min_img_size)]

    # Keep if already computed; otherwise compute
    if "class_area_px" not in df.columns:
        df["class_area_px"] = df["class_area_um2"] / (df["mpp_x"] * df["mpp_y"])

    if "frac_valid" not in df.columns:
        print("Computing frac_valid column")
        df["frac_valid"] = df["valid_pixels"] / (df["w"] * df["h"])

    df = df[df["frac_valid"] >= frac_valid_threshold]
    df = df[df["class_frac_valid"] >= area_threshold]
    df = df[df["class_frac_valid"] <= area_threshold_max]
    return df


def safe_sample(
    df: pd.DataFrame, n: int, replace: bool = False
) -> Optional[pd.DataFrame]:
    if len(df) < n:
        return None
    return df.sample(n=n, replace=replace)


_TILE_COLUMNS = {"origin_x", "origin_y", "w", "h"}


def build_crop_from_row(r: pd.Series) -> Optional[tuple[int, int, int, int]]:
    """If tile columns exist, build a PIL crop box (left, top, right, bottom); else None."""
    if not _TILE_COLUMNS.issubset(r.index):
        return None
    x = int(r["origin_x"])
    y = int(r["origin_y"])
    w = int(r["w"])
    h = int(r["h"])
    return (x, y, x + w, y + h)


def row_to_sample_ref(row: pd.Series) -> SampleRef:
    return SampleRef(
        dataset_id=row["dataset_id"],
        sample_id=row["sample_id"],
        image_relpath=row["image_relpath"],
        mask_relpath=row["mask_relpath"],
        class_id=int(row["dataset_class_id"]),
        crop=build_crop_from_row(row),
    )


def build_episodes_old(
    df_anorak: pd.DataFrame,
    n_ways: int,
    n_episodes_per_pair: int,
    n_shots: int,
    n_queries: int,
) -> Tuple[List[EpisodeRefs], Dict[Tuple[int, ...], int]]:
    n_classes = (
        df_anorak["dataset_class_id"].nunique() - 1
    )  # remove background class (0)

    episodes: List[EpisodeRefs] = []
    counts_by_way: Dict[Tuple[int, ...], int] = defaultdict(int)

    dataset_id = df_anorak["dataset_id"].iloc[0]
    dataset_dir = df_anorak["dataset_dir"].iloc[0]

    for way_classes in itertools.combinations(range(1, n_classes + 1), n_ways):
        used_sample_ids = set()
        for _ in range(n_episodes_per_pair):
            support_refs: List[SampleRef] = []
            query_refs: List[SampleRef] = []
            ok = True

            # --- SUPPORT ---
            for c in way_classes:
                pool = df_anorak[
                    (df_anorak["dataset_class_id"] == c)
                    & (~df_anorak["sample_id"].isin(used_sample_ids))
                ]
                s = safe_sample(pool, n=n_shots, replace=False)
                if s is None:
                    ok = False
                    break

                for _, row in s.iterrows():
                    ref = row_to_sample_ref(row)
                    support_refs.append(ref)
                    used_sample_ids.add(ref.sample_id)

            if not ok:
                warnings.warn(f"Skip episode (support impossible) for {way_classes}")
                continue

            # --- QUERY ---
            for c in way_classes:
                for _ in range(n_queries):
                    pool = df_anorak[
                        (df_anorak["dataset_class_id"] == c)
                        & (~df_anorak["sample_id"].isin(used_sample_ids))
                    ]
                    q = safe_sample(pool, n=1, replace=False)
                    if q is None:
                        ok = False
                        break

                    for _, row in q.iterrows():
                        ref = row_to_sample_ref(row)
                        query_refs.append(ref)
                        used_sample_ids.add(ref.sample_id)

                if not ok:
                    break

            if not ok:
                warnings.warn(f"Skip episode (query impossible) for {way_classes}")
                continue

            episodes.append(
                EpisodeRefs(
                    dataset_id=dataset_id,
                    dataset_dir=dataset_dir,
                    class_ids=list(way_classes),
                    support=support_refs,
                    query=query_refs,
                )
            )
            counts_by_way[tuple(way_classes)] += 1

    return episodes, counts_by_way


def build_episodes(
    df_anorak: pd.DataFrame,
    n_episodes_per_pair: int,
    n_shots: int,
    n_queries: int,
    seed: int = 0,
    n_ways: int = 2,
    way_classes_to_use: Optional[List[Tuple[int, ...]]] = None,
) -> Tuple[List[EpisodeRefs], Dict[Tuple[int, ...], int]]:
    # Keep it simple + explicit for fair testing
    if not (n_shots == 1 and n_queries == 1):
        raise ValueError(
            "This fair (unordered-pairs) builder currently supports only n_shots=1 and n_queries=1."
        )

    # Remove background class (0)
    class_ids = sorted([c for c in df_anorak["dataset_class_id"].unique() if c != 0])

    episodes: List[EpisodeRefs] = []
    counts_by_way: Dict[Tuple[int, ...], int] = defaultdict(int)

    dataset_id = df_anorak["dataset_id"].iloc[0]
    dataset_dir = df_anorak["dataset_dir"].iloc[0]

    if way_classes_to_use is not None:
        way_classes_list = [tuple(wc) for wc in way_classes_to_use]
    else:
        way_classes_list = list(itertools.combinations(class_ids, n_ways))

    for way_classes in way_classes_list:
        if way_classes_to_use is not None and way_classes not in way_classes_to_use:
            continue
        # For each class in this way-set, enumerate ALL unordered tile pairs once.
        # Each pair is a tuple of dataframe indices (i, j) with i < j.
        pairs_by_class: Dict[int, List[Tuple[int, int]]] = {}

        for c in way_classes:
            df_c = df_anorak[df_anorak["dataset_class_id"] == c]

            # Enforce support/query from different sample_id within a class
            # by removing pairs where sample_id matches.
            idx = df_c.index.to_list()
            all_pairs = []
            for i, j in itertools.combinations(idx, 2):
                if df_anorak.at[i, "sample_id"] != df_anorak.at[j, "sample_id"]:
                    all_pairs.append((i, j))

            if len(all_pairs) == 0:
                warnings.warn(
                    f"No valid (distinct sample_id) unordered pairs for class {c} in way_classes={way_classes}."
                )
                pairs_by_class[c] = []
                continue

            # Shuffle pairs ONCE for randomness, but we will NOT reuse them across episodes.
            # (Fair, but not deterministic ordering unless you set seed.)
            pairs_by_class[c] = (
                pd.Series(all_pairs).sample(frac=1.0, random_state=seed).tolist()
            )

        # How many episodes can we make without reusing pairs in ANY class?
        max_possible = (
            min(len(pairs_by_class[c]) for c in way_classes)
            if all(pairs_by_class[c] for c in way_classes)
            else 0
        )
        n_to_make = min(n_episodes_per_pair, max_possible)

        if n_to_make < n_episodes_per_pair:
            warnings.warn(
                f"way_classes={way_classes}: requested {n_episodes_per_pair} episodes, "
                f"but only {n_to_make} possible without reusing unordered pairs."
            )

        for ep_idx in range(n_to_make):
            support_refs: List[SampleRef] = []
            query_refs: List[SampleRef] = []

            # Build one episode by taking the next unused pair from each class
            for c in way_classes:
                i, j = pairs_by_class[c][ep_idx]  # each used once

                # Deterministic orientation (Option A): i -> support, j -> query
                support_refs.append(row_to_sample_ref(df_anorak.loc[i]))
                query_refs.append(row_to_sample_ref(df_anorak.loc[j]))

            episodes.append(
                EpisodeRefs(
                    dataset_id=dataset_id,
                    dataset_dir=dataset_dir,
                    class_ids=list(way_classes),
                    support=support_refs,
                    query=query_refs,
                )
            )
            counts_by_way[tuple(way_classes)] += 1

    return episodes, counts_by_way


def main() -> None:
    load_dotenv()

    data_root = Path(os.getenv("DATA_ROOT", "../data/")).resolve()
    fss_data_root = Path(os.getenv("FSS_DATA_ROOT", "../data/fss")).resolve()

    parquet_path = data_root / "index/tile_index_t896_s896/tile_index_t896_s896.parquet"
    df = pd.read_parquet(parquet_path)

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
    #
    if "class_area_px" not in df.columns:
        df["class_area_px"] = df["class_area_um2"] / (df["mpp_x"] * df["mpp_y"])

    if "frac_valid" not in df.columns:
        print("Computing frac_valid column")
        df["frac_valid"] = df["valid_pixels"] / (df["w"] * df["h"])

    output_dir = fss_data_root / "splits/scenario_anorak_2/"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    df_filtered = df[~df["is_border_tile"]]
    df_filtered = filter_index(
        df_filtered,
        area_threshold=0.25,
        area_threshold_max=0.75,
        min_img_size=896,
        frac_valid_threshold=0.85,
    )
    df_anorak = df_filtered[df_filtered["dataset_id"] == "anorak"]
    # adding class 3 and 5 of anorak to training, rest to test/val
    mask_class_test = df_anorak["dataset_class_id"].isin([4, 6])

    anorak_ids_test = df_anorak[mask_class_test]["sample_id"].unique().tolist()
    df_anorak_train = df_anorak[~df_anorak["sample_id"].isin(anorak_ids_test)]
    anorak_ids_train = df_anorak_train["sample_id"].unique().tolist()
    df_anorak_test = df_anorak[~df_anorak["sample_id"].isin(anorak_ids_train)]


    # --- episode params ---
    n_ways = 2
    n_episodes_per_pair = 100
    n_shots = 1
    n_queries = 1

    episodes, counts_by_way = build_episodes(
        df_anorak=df_anorak_test,
        n_ways=n_ways,
        n_episodes_per_pair=n_episodes_per_pair,
        n_shots=n_shots,
        n_queries=n_queries,
        way_classes_to_use=[(4, 6)],
    )

    save_path = output_dir / "test_episodes.json"
    save_episodes_json(save_path, episodes)
    print(f"Saved: {save_path}")

    # --- simple stats ---
    print("\nEpisodes per pair:")
    for way, cnt in sorted(counts_by_way.items()):
        print(f"  {way}: {cnt}/{n_episodes_per_pair}")
    print(f"\nTotal episodes: {len(episodes)}")

    # --- episode params ---
    n_ways = 2
    n_episodes_per_pair = 50
    n_shots = 1
    n_queries = 1

    episodes, counts_by_way = build_episodes(
        df_anorak=df_anorak_test,
        n_ways=n_ways,
        n_episodes_per_pair=n_episodes_per_pair,
        n_shots=n_shots,
        n_queries=n_queries,
        way_classes_to_use=[(4, 6)],
    )

    save_path = output_dir / "val_episodes.json"
    save_episodes_json(save_path, episodes)
    print(f"Saved: {save_path}")

    df_split.loc[df_split["dataset_id"] == "anorak", "split"] = "val"
    df_split.loc[df_split["dataset_id"] != "anorak", "split"] = "train"
    df_split.loc[df_split["sample_id"].isin(anorak_ids_train), "split"] = "train"

    df_split.to_csv(
        output_dir / "split.csv",
        index=False,
    )

    # --- simple stats ---
    print("\nEpisodes per pair:")
    for way, cnt in sorted(counts_by_way.items()):
        print(f"  {way}: {cnt}/{n_episodes_per_pair}")
    print(f"\nTotal episodes: {len(episodes)}")

    df_train = df[df["sample_id"].isin(df_split[df_split["split"] == "train"]["sample_id"])]
    # keeping only tile for class 3 and 5 of anorak in training, rest in test/val
    df_train = df_train[
        (df_train["dataset_id"] != "anorak")
        | ((df_train["dataset_class_id"].isin([3, 5])) & (df_train["dataset_id"] == "anorak"))
    ]
    tile_index_output_path = output_dir / "tile_index_train.parquet"
    df_train.to_parquet(tile_index_output_path)


if __name__ == "__main__":
    main()
