from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubjectPreferenceData:
    subject_id: int
    x: np.ndarray
    winner_indices: np.ndarray
    loser_indices: np.ndarray
    initial_targets: np.ndarray | None = None
    label: str | None = None
    dataset_name: str | None = None

    @property
    def comparison_count(self) -> int:
        return int(self.winner_indices.shape[0])

    @property
    def stimulus_count(self) -> int:
        return int(self.x.shape[0])


def standardize_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    scale = float(values.std())
    if scale <= 1e-12:
        return values - values.mean()
    return (values - values.mean()) / scale


def empirical_pairwise_scores(
    winner_indices: np.ndarray,
    loser_indices: np.ndarray,
    stimulus_count: int,
) -> np.ndarray:
    wins = np.zeros(stimulus_count, dtype=np.float64)
    losses = np.zeros(stimulus_count, dtype=np.float64)
    np.add.at(wins, winner_indices, 1.0)
    np.add.at(losses, loser_indices, 1.0)
    return standardize_vector(wins - losses)
