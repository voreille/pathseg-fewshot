from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class EpisodeSpec:
    ways: list[int]  # N
    shots: int  # K
    queries: int  # Q
    allow_background: bool = False  # whether class_id=0 can be sampled as a "way"


@dataclass(frozen=True)
class SampleRef:
    dataset_id: str
    sample_id: str
    image_relpath: str
    mask_relpath: str
    class_id: int
    crop: Optional[tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class EpisodeRefs:
    dataset_id: str
    dataset_dir: str
    class_ids: List[int]
    support: List[SampleRef]
    query: List[SampleRef]

    def get_sample_ids(self) -> List[str]:
        ids = [ref.sample_id for ref in self.support] + [ref.sample_id for ref in self.query]
        return list(set(ids))


def save_episodes_json(path: Path, episodes: Sequence[EpisodeRefs]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for e in episodes:
        d = {
            "dataset_id": e.dataset_id,
            "dataset_dir": e.dataset_dir,
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
                    "crop": r.crop,
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
                    "crop": r.crop,
                }
            )
        payload.append(d)

    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_episodes_json(path: Path | str) -> List[EpisodeRefs]:
    path = Path(path).resolve()
    with path.open("r") as f:
        data = json.load(f)

    episodes: List[EpisodeRefs] = []
    for e in data:
        support = [
            SampleRef(
                dataset_id=r["dataset_id"],
                sample_id=r["sample_id"],
                image_relpath=r["image_relpath"],
                mask_relpath=r["mask_relpath"],
                class_id=int(r["class_id"]),
                crop=r.get("crop"),
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
                crop=r.get("crop"),
            )
            for r in e["query"]
        ]
        episodes.append(
            EpisodeRefs(
                dataset_id=e["dataset_id"],
                dataset_dir=e["dataset_dir"],
                class_ids=list(map(int, e["class_ids"])),
                support=support,
                query=query,
            )
        )
    return episodes
