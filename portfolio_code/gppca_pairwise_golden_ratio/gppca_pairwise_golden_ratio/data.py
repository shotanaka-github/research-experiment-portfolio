from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SubjectPreferenceData:
    subject_id: int
    x: np.ndarray
    scores: np.ndarray
    winner_indices: np.ndarray
    loser_indices: np.ndarray

    @property
    def comparison_count(self) -> int:
        return int(self.winner_indices.shape[0])


def _resolve_column(frame: pd.DataFrame, preferred: str, fallback_index: int) -> np.ndarray:
    if preferred in frame.columns:
        return frame[preferred].to_numpy(dtype=np.float64)
    return frame.iloc[:, fallback_index].to_numpy(dtype=np.float64)


def _build_pairwise_comparisons(
    scores: np.ndarray,
    min_score_gap: float,
    max_comparisons: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    winners: list[int] = []
    losers: list[int] = []
    gaps: list[float] = []

    for left_index in range(scores.shape[0] - 1):
        for right_index in range(left_index + 1, scores.shape[0]):
            score_gap = scores[left_index] - scores[right_index]
            if abs(score_gap) < min_score_gap:
                continue
            if score_gap > 0.0:
                winners.append(left_index)
                losers.append(right_index)
            else:
                winners.append(right_index)
                losers.append(left_index)
            gaps.append(abs(float(score_gap)))

    winner_array = np.asarray(winners, dtype=np.int64)
    loser_array = np.asarray(losers, dtype=np.int64)
    gap_array = np.asarray(gaps, dtype=np.float64)

    if max_comparisons is not None and winner_array.shape[0] > max_comparisons:
        probabilities = gap_array / gap_array.sum()
        selected = rng.choice(winner_array.shape[0], size=max_comparisons, replace=False, p=probabilities)
        selected.sort()
        winner_array = winner_array[selected]
        loser_array = loser_array[selected]

    return winner_array, loser_array


def load_pairwise_golden_ratio_data(
    data_dir: str | Path,
    num_subjects: int = 20,
    min_score_gap: float = 1.0,
    max_comparisons: int | None = 400,
    random_state: int = 0,
) -> list[SubjectPreferenceData]:
    """Load golden-ratio data and convert ratings into pairwise comparisons."""

    data_dir = Path(data_dir)
    rng = np.random.default_rng(random_state)
    subjects: list[SubjectPreferenceData] = []

    for subject_id in range(1, num_subjects + 1):
        file_path = data_dir / f"data{subject_id}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing data file: {file_path}")

        frame = pd.read_csv(file_path)
        x = _resolve_column(frame, "aspect_ratio", 0).reshape(-1, 1)
        scores = _resolve_column(frame, "beauty", 1)
        winners, losers = _build_pairwise_comparisons(
            scores=scores,
            min_score_gap=min_score_gap,
            max_comparisons=max_comparisons,
            rng=rng,
        )
        if winners.size == 0:
            raise ValueError(
                f"Subject {subject_id} produced no pairwise comparisons. "
                "Try lowering min_score_gap."
            )

        subjects.append(
            SubjectPreferenceData(
                subject_id=subject_id,
                x=x.astype(np.float64),
                scores=scores.astype(np.float64),
                winner_indices=winners,
                loser_indices=losers,
            )
        )

    return subjects
