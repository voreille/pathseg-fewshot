from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from pathseg_fewshot.datasets.episode import EpisodeRefs, EpisodeSpec, SampleRef

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

_TILE_COLUMNS = {"origin_x", "origin_y", "w", "h"}


def _check_class_index(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"class_index is missing required columns: {missing}")


def _build_crop_from_row(r: pd.Series) -> Optional[tuple[int, int, int, int]]:
    """If tile columns exist, build a PIL crop box (left, top, right, bottom); else None."""
    if not _TILE_COLUMNS.issubset(r.index):
        return None
    x = int(r["origin_x"])
    y = int(r["origin_y"])
    w = int(r["w"])
    h = int(r["h"])
    return (x, y, x + w, y + h)


def _build_pools(df: pd.DataFrame) -> Tuple[List[str], Dict[Tuple[str, int], List[int]]]:
    """(dataset_id, class_id) -> list of row indices, plus sorted dataset_ids."""
    dataset_ids = sorted(df["dataset_id"].astype(str).unique().tolist())

    pool_indices: Dict[Tuple[str, int], List[int]] = {}
    for (ds, cid), sub in df.groupby(["dataset_id", "dataset_class_id"], sort=False):
        pool_indices[(str(ds), int(cid))] = sub.index.astype(int).tolist()

    return dataset_ids, pool_indices


def _build_rows_by_sample(df: pd.DataFrame) -> Dict[Tuple[str, str], List[int]]:
    """(dataset_id, sample_id) -> list of row indices (for unique_by='sample')."""
    out: Dict[Tuple[str, str], List[int]] = {}
    for (ds, sid), sub in df.groupby(["dataset_id", "sample_id"], sort=False):
        out[(str(ds), str(sid))] = sub.index.astype(int).tolist()
    return out


def _available_indices(pool: List[int], used_rows: Optional[set[int]]) -> List[int]:
    if not used_rows:
        return pool
    return [i for i in pool if i not in used_rows]


def _sample_k(rng: random.Random, pool: List[int], k: int) -> Optional[List[int]]:
    if len(pool) < k:
        return None
    return rng.sample(pool, k=k)


def _group_by_sample_id(df: pd.DataFrame, indices: List[int]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for idx in indices:
        sid = str(df.at[idx, "sample_id"])
        out.setdefault(sid, []).append(int(idx))
    return out


def _greedy_pack_tiles(
    *,
    rng: random.Random,
    df: pd.DataFrame,
    indices: List[int],
    k: int,
    forbidden_sample_ids: set[str],
) -> Optional[List[int]]:
    """
    Greedily pick tiles from as few sample_ids as possible.
    Returns k indices or None.
    """
    by_sid = _group_by_sample_id(df, indices)

    # remove forbidden
    for sid in list(by_sid.keys()):
        if sid in forbidden_sample_ids:
            by_sid.pop(sid, None)

    # sort sample_ids by how many tiles they offer (descending)
    sids = sorted(by_sid.keys(), key=lambda s: len(by_sid[s]), reverse=True)

    chosen: List[int] = []
    for sid in sids:
        if len(chosen) >= k:
            break
        tiles = by_sid[sid]
        rng.shuffle(tiles)
        take = min(k - len(chosen), len(tiles))
        chosen.extend(tiles[:take])

    if len(chosen) < k:
        return None
    return chosen


def _rows_to_episode(
    *,
    df: pd.DataFrame,
    dataset_id: str,
    class_ids: List[int],
    support_idx: List[int],
    query_idx: List[int],
) -> EpisodeRefs:
    def to_ref(idx: int) -> SampleRef:
        r = df.loc[idx]
        return SampleRef(
            dataset_id=str(r["dataset_id"]),
            sample_id=str(r["sample_id"]),
            image_relpath=str(r["image_relpath"]),
            mask_relpath=str(r["mask_relpath"]),
            class_id=int(r["dataset_class_id"]),
            crop=_build_crop_from_row(r),
        )

    support = [to_ref(i) for i in support_idx]
    query = [to_ref(i) for i in query_idx]

    # dataset_dir should be constant within dataset; pick from any row if possible
    if support_idx:
        dataset_dir = str(df.loc[support_idx[0], "dataset_dir"])
    elif query_idx:
        dataset_dir = str(df.loc[query_idx[0], "dataset_dir"])
    else:
        dataset_dir = ""

    return EpisodeRefs(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        class_ids=class_ids,
        support=support,
        query=query,
    )


# -----------------------
# Shared sampling logic
# -----------------------

@dataclass(frozen=True)
class _EpisodePlan:
    ds_id: str
    class_ids: List[int]
    ways: int


class EpisodeSamplerBase:
    """
    Public contract:
      - sample_episode(...) ALWAYS returns EpisodeRefs
      - raises ValueError if it cannot sample within max_tries
    """

    def sample_episode(
        self,
        *,
        seed: int,
        dataset_id: Optional[str] = None,
        max_tries: int = 50,
        allow_background: Optional[bool] = None,
    ) -> EpisodeRefs:
        ep = self._try_sample_episode(
            seed=seed,
            dataset_id=dataset_id,
            max_tries=max_tries,
            allow_background=allow_background,
        )
        if ep is None:
            raise ValueError(
                "Failed to sample episode after max_tries. "
                "Consider reducing ways/shots/queries, changing unique_by, or resetting the sampler."
            )
        return ep

    def _try_sample_episode(
        self,
        *,
        seed: int,
        dataset_id: Optional[str],
        max_tries: int,
        allow_background: Optional[bool],
    ) -> Optional[EpisodeRefs]:
        raise NotImplementedError


class _EpisodeSamplerCommon:
    """
    Common utilities for samplers.
    Expects subclasses to define:
      - self.df: pd.DataFrame
      - self.spec: EpisodeSpec
      - self.dataset_ids: List[str]
      - self._pool_indices: Dict[(ds_id, class_id), List[int]]
    """

    df: pd.DataFrame
    spec: EpisodeSpec
    dataset_ids: List[str]
    _pool_indices: Dict[Tuple[str, int], List[int]]

    def _available_classes(self, ds_id: str, *, allow_background: bool) -> List[int]:
        return sorted(
            {
                cid
                for (ds, cid) in self._pool_indices.keys()
                if ds == ds_id and (allow_background or cid != 0)
            }
        )

    def _choose_plan(
        self,
        *,
        rng: random.Random,
        dataset_id: Optional[str],
        allow_background: Optional[bool] = None,
    ) -> _EpisodePlan:
        spec = self.spec
        ways = int(rng.choice(spec.ways))

        ds_id = str(dataset_id) if dataset_id is not None else rng.choice(self.dataset_ids)

        allow_bg = spec.allow_background if allow_background is None else bool(allow_background)
        classes = self._available_classes(ds_id, allow_background=allow_bg)

        if len(classes) < ways:
            raise ValueError(
                f"Not enough classes (n={len(classes)}) to sample {ways}-way episode "
                f"from dataset '{ds_id}' (allow_background={allow_bg}, spec={spec})."
            )

        class_ids = rng.sample(classes, k=ways)
        return _EpisodePlan(ds_id=ds_id, class_ids=[int(x) for x in class_ids], ways=ways)


# -----------------------
# Stateless sampler (train)
# -----------------------

class StatelessEpisodeSampler(EpisodeSamplerBase, _EpisodeSamplerCommon):
    """
    Stateless sampling:
      - deterministic given seed
      - safe in DataLoader workers
      - no 'used' tracking
    """

    def __init__(self, class_index: pd.DataFrame, spec: EpisodeSpec) -> None:
        _check_class_index(class_index)

        df = class_index.copy()
        df["dataset_id"] = df["dataset_id"].astype(str)
        df["sample_id"] = df["sample_id"].astype(str)
        df["dataset_class_id"] = df["dataset_class_id"].astype(int)

        self.df = df
        self.spec = spec
        self.dataset_ids, self._pool_indices = _build_pools(self.df)

    def _try_sample_episode(
        self,
        *,
        seed: int,
        dataset_id: Optional[str],
        max_tries: int,
        allow_background: Optional[bool],
    ) -> Optional[EpisodeRefs]:
        rng = random.Random(int(seed))
        spec = self.spec
        needed = int(spec.shots + spec.queries)

        for _ in range(int(max_tries)):
            plan = self._choose_plan(
                rng=rng,
                dataset_id=dataset_id,
                allow_background=allow_background,
            )

            support_idx: List[int] = []
            query_idx: List[int] = []
            ok = True

            for cid in plan.class_ids:
                pool = self._pool_indices.get((plan.ds_id, int(cid)), [])
                chosen = _sample_k(rng, pool, needed)
                if chosen is None:
                    ok = False
                    break
                support_idx.extend(chosen[: spec.shots])
                query_idx.extend(chosen[spec.shots:])

            if not ok:
                continue

            return _rows_to_episode(
                df=self.df,
                dataset_id=plan.ds_id,
                class_ids=plan.class_ids,
                support_idx=support_idx,
                query_idx=query_idx,
            )

        return None


# -----------------------
# Consuming sampler (build banks)
# -----------------------

class ConsumingEpisodeSampler(EpisodeSamplerBase, _EpisodeSamplerCommon):
    """
    Stateful sampler enforcing no overlap across produced episodes.

    unique_by:
      - "row":    uniqueness at tile-row level (recommended for tile indices)
      - "sample": uniqueness at image level (very restrictive for tile indices)
    """

    def __init__(
        self,
        class_index: pd.DataFrame,
        spec: EpisodeSpec,
        *,
        unique_by: str = "row",
    ) -> None:
        if unique_by not in {"row", "sample"}:
            raise ValueError("unique_by must be 'row' or 'sample'")

        _check_class_index(class_index)

        df = class_index.copy()
        df["dataset_id"] = df["dataset_id"].astype(str)
        df["sample_id"] = df["sample_id"].astype(str)
        df["dataset_class_id"] = df["dataset_class_id"].astype(int)

        self.df = df
        self.spec = spec
        self.unique_by = unique_by

        self.dataset_ids, self._pool_indices = _build_pools(self.df)
        self._rows_by_sample = _build_rows_by_sample(self.df) if unique_by == "sample" else {}
        self._used_rows: set[int] = set()

    def reset_used(self) -> None:
        self._used_rows.clear()

    def _mark_used_rows(self, picked_row_indices: List[int]) -> None:
        if self.unique_by == "row":
            self._used_rows.update(int(i) for i in picked_row_indices)
            return

        # unique_by == "sample": mark all rows belonging to the same (dataset_id, sample_id)
        for idx in picked_row_indices:
            ds = str(self.df.at[idx, "dataset_id"])
            sid = str(self.df.at[idx, "sample_id"])
            for rid in self._rows_by_sample.get((ds, sid), []):
                self._used_rows.add(int(rid))

    def _try_sample_episode(
        self,
        *,
        seed: int,
        dataset_id: Optional[str],
        max_tries: int,
        allow_background: Optional[bool],
    ) -> Optional[EpisodeRefs]:
        rng = random.Random(int(seed))
        spec = self.spec
        needed = int(spec.shots + spec.queries)

        for _ in range(int(max_tries)):
            plan = self._choose_plan(
                rng=rng,
                dataset_id=dataset_id,
                allow_background=allow_background,
            )

            support_idx: List[int] = []
            query_idx: List[int] = []
            ok = True

            for cid in plan.class_ids:
                base_pool = self._pool_indices.get((plan.ds_id, int(cid)), [])
                avail = _available_indices(base_pool, self._used_rows)

                chosen = _sample_k(rng, avail, needed)
                if chosen is None:
                    ok = False
                    break

                support_idx.extend(chosen[: spec.shots])
                query_idx.extend(chosen[spec.shots:])

            if not ok:
                continue

            self._mark_used_rows(support_idx)
            self._mark_used_rows(query_idx)

            return _rows_to_episode(
                df=self.df,
                dataset_id=plan.ds_id,
                class_ids=plan.class_ids,
                support_idx=support_idx,
                query_idx=query_idx,
            )

        return None

    def build_episode_list(
        self,
        *,
        episodes_per_dataset: int,
        seed: int = 0,
        dataset_ids: Optional[Sequence[str]] = None,
        max_tries_per_episode: int = 200,
        allow_background: Optional[bool] = None,
    ) -> List[EpisodeRefs]:
        if dataset_ids is None:
            dataset_ids = self.dataset_ids

        bank: List[EpisodeRefs] = []
        for ds in dataset_ids:
            for j in range(int(episodes_per_dataset)):
                ep_seed = (int(seed) * 10_000) + (hash(ds) % 1_000) * 100 + j
                ep = self.sample_episode(
                    seed=ep_seed,
                    dataset_id=str(ds),
                    max_tries=int(max_tries_per_episode),
                    allow_background=allow_background,
                )
                bank.append(ep)
        return bank


# -----------------------
# Min-images consuming sampler (packs tiles into few images)
# -----------------------

class MinImagesConsumingEpisodeSampler(ConsumingEpisodeSampler):
    """
    Like ConsumingEpisodeSampler, but tries to use as few sample_id as possible.

    Additional constraint (per class):
      - support and query are sampled from disjoint images (no shared sample_id)

    Notes:
      - Packs support tiles into 1 (or few) images if possible,
        and packs query tiles into 1 (or few) *different* images.
    """

    def _try_sample_episode(
        self,
        *,
        seed: int,
        dataset_id: Optional[str],
        max_tries: int,
        allow_background: Optional[bool],
    ) -> Optional[EpisodeRefs]:
        rng = random.Random(int(seed))
        spec = self.spec

        for _ in range(int(max_tries)):
            plan = self._choose_plan(
                rng=rng,
                dataset_id=dataset_id,
                allow_background=allow_background,
            )

            support_idx: List[int] = []
            query_idx: List[int] = []
            ok_all_classes = True

            for cid in plan.class_ids:
                base_pool = self._pool_indices.get((plan.ds_id, int(cid)), [])
                avail = _available_indices(base_pool, self._used_rows)

                # 1) pick support (pack into few images)
                support_for_class = _greedy_pack_tiles(
                    rng=rng,
                    df=self.df,
                    indices=avail,
                    k=int(spec.shots),
                    forbidden_sample_ids=set(),
                )
                if support_for_class is None:
                    logging.info(
                        "Failed to pack support tiles for class %s in dataset %s; retrying...",
                        cid,
                        plan.ds_id,
                    )
                    ok_all_classes = False
                    break

                support_sids = {str(self.df.at[i, "sample_id"]) for i in support_for_class}

                # 2) pick query from different images than support (pack as well)
                remaining = [
                    i for i in avail if str(self.df.at[i, "sample_id"]) not in support_sids
                ]
                query_for_class = _greedy_pack_tiles(
                    rng=rng,
                    df=self.df,
                    indices=remaining,
                    k=int(spec.queries),
                    forbidden_sample_ids=set(),
                )
                if query_for_class is None:
                    logging.info(
                        "Failed to pack query tiles for class %s in dataset %s; retrying...",
                        cid,
                        plan.ds_id,
                    )
                    ok_all_classes = False
                    break

                support_idx.extend(support_for_class)
                query_idx.extend(query_for_class)

            if not ok_all_classes:
                continue

            self._mark_used_rows(support_idx)
            self._mark_used_rows(query_idx)

            return _rows_to_episode(
                df=self.df,
                dataset_id=plan.ds_id,
                class_ids=plan.class_ids,
                support_idx=support_idx,
                query_idx=query_idx,
            )

        return None
