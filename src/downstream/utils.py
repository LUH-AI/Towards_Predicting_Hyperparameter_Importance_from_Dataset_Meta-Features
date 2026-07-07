import json
from pathlib import Path

import numpy as np


def json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=json_default)


def normalize_hpi_dict(hpi: dict[str, float], hp_names: list[str]) -> dict[str, float]:
    vals = np.array([max(0.0, float(hpi.get(hp, 0.0))) for hp in hp_names], dtype=float)
    s = vals.sum()

    if s <= 0:
        vals = np.ones(len(hp_names), dtype=float) / len(hp_names)
    else:
        vals = vals / s

    return {hp: float(v) for hp, v in zip(hp_names, vals)}


def topk_from_scores(score_dict: dict[str, float], k: int) -> list[str]:
    return sorted(score_dict.keys(), key=lambda hp: score_dict[hp], reverse=True)[:k]