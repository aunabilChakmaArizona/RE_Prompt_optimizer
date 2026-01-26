from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Optional


DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "data")
)


def read_json_file(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_pickle_file(file_path: str) -> Any:
    with open(file_path, "rb") as handle:
        return pickle.load(handle)


def load_train_samples(
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    filename: str = "fs_tacred_train_samples.pkl",
) -> Dict[str, Any]:
    return read_pickle_file(os.path.join(data_dir, filename))


def load_split_episodes(
    *,
    split: str,
    data_dir: str = DEFAULT_DATA_DIR,
    dataset_prefix: str = "fs_tacred",
    ep_start: int = 0,
    ep_end: Optional[int] = None,
) -> Dict[str, Any]:
    episodes_path = os.path.join(
        data_dir, f"{dataset_prefix}_{split}_episodes_1shots.pkl"
    )
    details_path = os.path.join(
        data_dir, f"{dataset_prefix}_{split}_episodes_shots_details.pkl"
    )

    episodes: List[Dict[str, Any]] = read_pickle_file(episodes_path)
    if ep_end is None:
        episodes = episodes[ep_start:]
    else:
        episodes = episodes[ep_start:ep_end]

    details = read_pickle_file(details_path)
    return {
        "episodes": episodes,
        "shots": details.get("shots", {}),
        "queries": details.get("queries", {}),
        "umbc_shots": details.get("umbc_shots", {})
    }
