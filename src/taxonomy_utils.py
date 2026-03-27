from __future__ import annotations

from pathlib import Path

import pandas as pd


KNOWN_MAJOR_CLASS_COLUMNS = [
    "cell_type",
    "broad_cell_type",
    "class",
    "supertype",
]
KNOWN_SUBCLASS_COLUMNS = [
    "subclass",
    "subtype",
    "cell_subclass",
]
KNOWN_CONFIDENCE_COLUMNS = [
    "annotation_confidence",
    "confidence",
    "mapping_score",
]


def attach_existing_taxonomy(obs: pd.DataFrame) -> pd.DataFrame:
    result = obs.copy()
    result["major_cell_class"] = _pick_first_available(obs, KNOWN_MAJOR_CLASS_COLUMNS, fill_value="unknown")
    result["subclass"] = _pick_first_available(obs, KNOWN_SUBCLASS_COLUMNS, fill_value="unknown")
    result["annotation_confidence"] = _pick_first_available(obs, KNOWN_CONFIDENCE_COLUMNS, fill_value=0.0)
    result["low_confidence_annotation"] = pd.to_numeric(result["annotation_confidence"], errors="coerce").fillna(0) < 0.5
    return result


def write_annotation_template(path: str | Path, obs_names: list[str]) -> Path:
    template = pd.DataFrame(
        {
            "obs_name": obs_names,
            "major_cell_class": "unknown",
            "subclass": "unknown",
            "annotation_confidence": 0.0,
        }
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(out, sep="\t", index=False)
    return out


def _pick_first_available(obs: pd.DataFrame, columns: list[str], fill_value):
    for column in columns:
        if column in obs.columns:
            return obs[column]
    return pd.Series([fill_value] * len(obs), index=obs.index)
