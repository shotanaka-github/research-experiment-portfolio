from __future__ import annotations

from pathlib import Path
from typing import Any

from .color_lab import load_color_lab_pairwise_data
from .golden_ratio import load_pairwise_golden_ratio_data
from .thurstone import load_thurstone_pairwise_data
from ..types import SubjectPreferenceData

DATASET_NAMES = (
    "thurstone_pairwise",
    "golden_ratio_induced",
    "color_lab_pairwise",
)


def load_dataset(
    dataset: str,
    data_dir: str | Path | None = None,
    **kwargs: Any,
) -> list[SubjectPreferenceData]:
    if dataset == "thurstone_pairwise":
        return load_thurstone_pairwise_data(data_dir=data_dir, **kwargs)
    if dataset == "golden_ratio_induced":
        return load_pairwise_golden_ratio_data(data_dir=data_dir, **kwargs)
    if dataset == "color_lab_pairwise":
        return load_color_lab_pairwise_data(data_dir=data_dir, **kwargs)
    raise ValueError(f"Unknown dataset: {dataset}")


__all__ = [
    "DATASET_NAMES",
    "SubjectPreferenceData",
    "load_color_lab_pairwise_data",
    "load_dataset",
    "load_pairwise_golden_ratio_data",
    "load_thurstone_pairwise_data",
]
