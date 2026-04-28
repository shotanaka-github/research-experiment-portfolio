from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..types import SubjectPreferenceData, empirical_pairwise_scores


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "thurstone" / "data10"


def load_thurstone_pairwise_data(
    data_dir: str | Path | None = None,
    session_count: int = 1,
    subject_count: int = 14,
) -> list[SubjectPreferenceData]:
    data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
    subjects: list[SubjectPreferenceData] = []
    running_subject_id = 1

    for session_id in range(1, session_count + 1):
        session_dir = data_dir / f"session{session_id}"
        if not session_dir.exists():
            raise FileNotFoundError(f"Missing session directory: {session_dir}")

        for subject_index in range(1, subject_count + 1):
            feature_file = session_dir / f"features_{subject_index}_converted.csv"
            pairwise_file = session_dir / f"pairwise_{subject_index}.csv"
            if not feature_file.exists() or not pairwise_file.exists():
                raise FileNotFoundError(f"Missing Thurstone files for subject {subject_index} in {session_dir}")

            x = pd.read_csv(feature_file, header=None).to_numpy(dtype=np.float64)
            pairwise = pd.read_csv(pairwise_file)
            left = pairwise["A_index"].to_numpy(dtype=np.int64)
            right = pairwise["B_index"].to_numpy(dtype=np.int64)
            chosen = pairwise["chosen"].to_numpy(dtype=np.int64)

            winners = np.where(chosen == 1, right, left)
            losers = np.where(chosen == 1, left, right)
            initial_targets = empirical_pairwise_scores(winners, losers, x.shape[0])

            subjects.append(
                SubjectPreferenceData(
                    subject_id=running_subject_id,
                    x=x,
                    winner_indices=winners,
                    loser_indices=losers,
                    initial_targets=initial_targets,
                    label=f"session{session_id}_subject{subject_index}",
                    dataset_name="thurstone_pairwise",
                )
            )
            running_subject_id += 1

    return subjects
