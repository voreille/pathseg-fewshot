# episode_sampler.py
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd


# -----------------------
# Dataclasses
# -----------------------

@dataclass(frozen=True)
class EpisodeSpec:
    ways: int  # N
    shots: int  # K
    queries: int  # Q
    crop_size: int = 448


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class WholeImage:
    """Sentinel crop spec meaning 'use whole image'."""
    pass


CropSpec = Union[CropBox, WholeImage]


@dataclass(frozen=True)
class SampleRef:
    dataset_id: str
    sample_id: str
    image_relpath: str
    mask_relpath: str
    class_id: int
    crop: Optional[CropSpec] = None  # None -> transform decides (e.g., random crop)


@dataclass(frozen=True)
class EpisodeRefs:
    dataset_id: str
    class_ids: List[int]          # length N
    support: List[SampleRef]      # length N*K
    query: List[SampleRef]        # length N*Q


# -----------------------
# Helpers
# -----------------------

_REQUIRED_COLUMNS = {
    "dataset_id",
    "dataset_class_id",
    "sample_id",
    "image_relpath",
    "mask_relpath",
}


def _check_class_index(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"class_index is missing required columns: {missing}")


def _build_indexes(
    df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, pd.DataFrame], Dict[Tuple[str, int], pd.DataFrame]]:
    """
    Returns:
      dataset_ids,
      by_dataset,
      by_dataset_class
    """
    df = df.copy()
    df["dataset_class_id"] = df["dataset_class_id"].astype(int)

    by_dataset: Dict[str, pd.DataFrame] = {}
    by_dataset_class: Dict[Tuple[str, int], pd.DataFrame] = {}

    for ds_id, sub in df.groupby("dataset_id"):
        by_dataset[str(ds_id)] = sub
        for cid, sub2 in sub.groupby("dataset_class_id"):
            by_dataset_class[(str(ds_id), int(cid))] = sub2

    dataset_ids = sorted(by_dataset.keys())
    if not dataset_ids:
        raise ValueError("No datasets available in class_index.")

    return dataset_ids, by_dataset, by_dataset_class


def _sample_episode_from_pools(
    *,
    rng: random.Random,
    spec: EpisodeSpec,
    dataset_id: str,
    by_dataset: Dict[str, pd.DataFrame],
    by_dataset_class: Dict[Tuple[str, int], pd.DataFrame],
    pool_filter_fn,  # (pool_df: pd.DataFrame) -> pd.DataFrame
) -> Optional[Tuple[List[int], List[Tuple[int, pd.Series]], List[Tuple[int, pd.Series]]]]:
    """
    Core episode construction logic. Shared by stateless and consuming samplers.

    Returns:
      (class_ids, support_rows, query_rows)
    where support_rows/query_rows are lists of (row_index, row_series).

    Returns None if an episode cannot be formed (e.g. not enough available rows).
    """
    ds_df = by_dataset[dataset_id]
    available_classes = sorted(ds_df["dataset_class_id"].unique().tolist())
    if len(available_classes) < spec.ways:
        return None

    class_ids = rng.sample(available_classes, k=spec.ways)
    needed = spec.shots + spec.queries

    support_rows: List[Tuple[int, pd.Series]] = []
    query_rows: List[Tuple[int, pd.Series]] = []

    for cid in class_ids:
        pool = by_dataset_class[(dataset_id, int(cid))]
        pool = pool_filter_fn(pool)
        if len(pool) < needed:
            return None

        chosen = pool.sample(n=needed, random_state=rng.randrange(10**9))
        # NOTE: iterrows preserves original index values (important for consuming sampler)
        for j, (row_idx, row) in enumerate(chosen.iterrows()):
            if j < spec.shots:
                support_rows.append((int(row_idx), row))
            else:
                query_rows.append((int(row_idx), row))

    return [int(x) for x in class_ids], support_rows, query_rows


def _rows_to_episode(
    *,
    dataset_id: str,
    class_ids: List[int],
    support_rows: List[Tuple[int, pd.Series]],
    query_rows: List[Tuple[int, pd.Series]],
    # optional crop assignment hook
    crop_for_row_fn=None,  # (row: pd.Series, split: "support"|"query", idx: int) -> Optional[CropSpec]
) -> EpisodeRefs:
    support: List[SampleRef] = []
    query: List[SampleRef] = []

    if crop_for_row_fn is None:
        def crop_for_row_fn(row, split, idx):  # noqa: ANN001
            return None

    for i, (_, r) in enumerate(support_rows):
        support.append(
            SampleRef(
                dataset_id=str(r["dataset_id"]),
                sample_id=str(r["sample_id"]),
                image_relpath=str(r["image_relpath"]),
                mask_relpath=str(r["mask_relpath"]),
                class_id=int(r["dataset_class_id"]) if False else int(r["dataset_class_id"])  # unused; kept for clarity
            )
        )

    # The class_id in SampleRef should be the episode-selected class, not necessarily r["dataset_class_id"]?
    # In your original design, row is already for that class; so it's safe to use row["dataset_class_id"].
    # But we *must* write class_id = the selected class id. We'll reconstruct properly:
    # We'll re-walk support_rows/query_rows grouped in the same order we sampled (by cid).
    #
    # To avoid ambiguity, we rebuild with row["dataset_class_id"] (same as cid).
    support = []
    for i, (_, r) in enumerate(support_rows):
        support.append(
            SampleRef(
                dataset_id=str(r["dataset_id"]),
                sample_id=str(r["sample_id"]),
                image_relpath=str(r["image_relpath"]),
                mask_relpath=str(r["mask_relpath"]),
                class_id=int(r["dataset_class_id"]),
                crop=crop_for_row_fn(r, "support", i),
            )
        )

    query = []
    for i, (_, r) in enumerate(query_rows):
        query.append(
            SampleRef(
                dataset_id=str(r["dataset_id"]),
                sample_id=str(r["sample_id"]),
                image_relpath=str(r["image_relpath"]),
                mask_relpath=str(r["mask_relpath"]),
                class_id=int(r["dataset_class_id"]),
                crop=crop_for_row_fn(r, "query", i),
            )
        )

    return EpisodeRefs(
        dataset_id=dataset_id,
        class_ids=class_ids,
        support=support,
        query=query,
    )


# -----------------------
# Base API
# -----------------------

class EpisodeSamplerBase:
    """
    Minimal “interface” for episode samplers.
    """
    def sample_episode(
        self,
        *,
        seed: int,
        dataset_id: Optional[str] = None,
        max_tries: int = 50,
    ) -> Optional[EpisodeRefs]:
        raise NotImplementedError


# -----------------------
# Stateless sampler (safe in DataLoader workers)
# -----------------------

class StatelessEpisodeSampler(EpisodeSamplerBase):
    """
    Stateless episode sampler:
      - safe with DataLoader(num_workers>0)
      - deterministic given seed
      - does NOT enforce no-overlap across episodes (that’s a split responsibility)
    """

    def __init__(self, class_index: pd.DataFrame, spec: EpisodeSpec) -> None:
        _check_class_index(class_index)
        self.df = class_index.copy()
        self.df["dataset_class_id"] = self.df["dataset_class_id"].astype(int)
        self.spec = spec
        self.dataset_ids, self._by_dataset, self._by_dataset_class = _build_indexes(self.df)

    def sample_episode(
        self,
        *,
        seed: int,
        dataset_id: Optional[str] = None,
        max_tries: int = 50,
    ) -> Optional[EpisodeRefs]:
        rng = random.Random(int(seed))

        for _ in range(int(max_tries)):
            ds_id = dataset_id if dataset_id is not None else rng.choice(self.dataset_ids)
            if ds_id not in self._by_dataset:
                raise ValueError(f"Unknown dataset_id={ds_id}")

            out = _sample_episode_from_pools(
                rng=rng,
                spec=self.spec,
                dataset_id=ds_id,
                by_dataset=self._by_dataset,
                by_dataset_class=self._by_dataset_class,
                pool_filter_fn=lambda pool: pool,  # no filtering
            )
            if out is None:
                if dataset_id is not None:
                    return None
                continue

            class_ids, support_rows, query_rows = out
            return _rows_to_episode(
                dataset_id=ds_id,
                class_ids=class_ids,
                support_rows=support_rows,
                query_rows=query_rows,
            )

        return None


# -----------------------
# Consuming sampler (for building fixed banks)
# -----------------------

class ConsumingEpisodeSampler(EpisodeSamplerBase):
    """
    Stateful sampler enforcing no-overlap across produced episodes.

    NOTE: Not safe to rely on this for uniqueness when used inside a multi-worker DataLoader,
    because each worker will have its own instance/state. Use it offline (single process) to
    build an episode bank, then serialize it for val/test.

    unique_by:
      - "sample": uniqueness by (dataset_id, sample_id) across all episodes (recommended)
      - "row":    uniqueness by row index in class_index
    """

    def __init__(
        self,
        class_index: pd.DataFrame,
        spec: EpisodeSpec,
        *,
        unique_by: str = "sample",
    ) -> None:
        if unique_by not in {"sample", "row"}:
            raise ValueError("unique_by must be 'sample' or 'row'")

        _check_class_index(class_index)
        self.df = class_index.copy()
        self.df["dataset_class_id"] = self.df["dataset_class_id"].astype(int)
        self.spec = spec
        self.unique_by = unique_by

        self.dataset_ids, self._by_dataset, self._by_dataset_class = _build_indexes(self.df)

        self._used_samples: set[Tuple[str, str]] = set()
        self._used_rows: set[int] = set()

    def reset_used(self) -> None:
        self._used_samples.clear()
        self._used_rows.clear()

    def _pool_filter(self, pool: pd.DataFrame) -> pd.DataFrame:
        if self.unique_by == "row":
            if not self._used_rows:
                return pool
            return pool.loc[~pool.index.isin(self._used_rows)]

        # unique_by == "sample"
        if not self._used_samples:
            return pool

        used = self._used_samples
        keys = list(zip(pool["dataset_id"].astype(str), pool["sample_id"].astype(str)))
        mask = [k not in used for k in keys]
        return pool.loc[mask]

    def _mark_used(self, row_idx: int, row: pd.Series) -> None:
        if self.unique_by == "row":
            self._used_rows.add(int(row_idx))
        else:
            self._used_samples.add((str(row["dataset_id"]), str(row["sample_id"])))

    def sample_episode(
        self,
        *,
        seed: int,
        dataset_id: Optional[str] = None,
        max_tries: int = 50,
    ) -> Optional[EpisodeRefs]:
        rng = random.Random(int(seed))

        for _ in range(int(max_tries)):
            ds_id = dataset_id if dataset_id is not None else rng.choice(self.dataset_ids)
            if ds_id not in self._by_dataset:
                raise ValueError(f"Unknown dataset_id={ds_id}")

            out = _sample_episode_from_pools(
                rng=rng,
                spec=self.spec,
                dataset_id=ds_id,
                by_dataset=self._by_dataset,
                by_dataset_class=self._by_dataset_class,
                pool_filter_fn=self._pool_filter,
            )
            if out is None:
                if dataset_id is not None:
                    return None
                continue

            class_ids, support_rows, query_rows = out

            # commit usage
            for row_idx, row in support_rows:
                self._mark_used(row_idx, row)
            for row_idx, row in query_rows:
                self._mark_used(row_idx, row)

            return _rows_to_episode(
                dataset_id=ds_id,
                class_ids=class_ids,
                support_rows=support_rows,
                query_rows=query_rows,
            )

        return None

    def build_episode_list(
        self,
        *,
        episodes_per_dataset: int,
        seed: int = 0,
        dataset_ids: Optional[Sequence[str]] = None,
    ) -> List[EpisodeRefs]:
        if dataset_ids is None:
            dataset_ids = self.dataset_ids

        bank: List[EpisodeRefs] = []
        for ds in dataset_ids:
            for j in range(int(episodes_per_dataset)):
                ep_seed = (int(seed) * 10_000) + (hash(ds) % 1_000) * 100 + j
                ep = self.sample_episode(seed=ep_seed, dataset_id=ds, max_tries=100)
                if ep is None:
                    break
                bank.append(ep)
        return bank


# -----------------------
# Serialization
# -----------------------

def save_episodes_json(path: Path, episodes: Sequence[EpisodeRefs]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def encode_crop(c: Optional[CropSpec]):
        if c is None:
            return None
        if isinstance(c, WholeImage):
            return {"type": "whole"}
        if isinstance(c, CropBox):
            return {"type": "box", **asdict(c)}
        raise TypeError(f"Unknown crop type: {type(c)}")

    payload = []
    for e in episodes:
        d = {
            "dataset_id": e.dataset_id,
            "class_ids": list(map(int, e.class_ids)),
            "support": [],
            "query": [],
        }
        for r in e.support:
            d["support"].append(
                {
                    "dataset_id": r.dataset_id,
                    "sample_id": r.sample_id,
                    "image_relpath": r.image_relpath,
                    "mask_relpath": r.mask_relpath,
                    "class_id": int(r.class_id),
                    "crop": encode_crop(r.crop),
                }
            )
        for r in e.query:
            d["query"].append(
                {
                    "dataset_id": r.dataset_id,
                    "sample_id": r.sample_id,
                    "image_relpath": r.image_relpath,
                    "mask_relpath": r.mask_relpath,
                    "class_id": int(r.class_id),
                    "crop": encode_crop(r.crop),
                }
            )
        payload.append(d)

    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_episodes_json(path: Path) -> List[EpisodeRefs]:
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)

    def decode_crop(c):
        if c is None:
            return None
        t = c.get("type")
        if t == "whole":
            return WholeImage()
        if t == "box":
            return CropBox(
                left=int(c["left"]),
                top=int(c["top"]),
                right=int(c["right"]),
                bottom=int(c["bottom"]),
            )
        raise ValueError(f"Invalid crop payload: {c}")

    episodes: List[EpisodeRefs] = []
    for e in data:
        support = [
            SampleRef(
                dataset_id=r["dataset_id"],
                sample_id=r["sample_id"],
                image_relpath=r["image_relpath"],
                mask_relpath=r["mask_relpath"],
                class_id=int(r["class_id"]),
                crop=decode_crop(r.get("crop")),
            )
            for r in e["support"]
        ]
        query = [
            SampleRef(
                dataset_id=r["dataset_id"],
                sample_id=r["sample_id"],
                image_relpath=r["image_relpath"],
                mask_relpath=r["mask_relpath"],
                class_id=int(r["class_id"]),
                crop=decode_crop(r.get("crop")),
            )
            for r in e["query"]
        ]
        episodes.append(
            EpisodeRefs(
                dataset_id=e["dataset_id"],
                class_ids=list(map(int, e["class_ids"])),
                support=support,
                query=query,
            )
        )
    return episodes
