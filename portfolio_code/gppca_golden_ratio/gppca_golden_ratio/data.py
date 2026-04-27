from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_golden_ratio_data(data_dir: str | Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load the 20-subject rectangle preference dataset.

    Each CSV is expected to contain at least three columns. The original project
    uses the third column as the rectangle aspect ratio and the second column as
    the evaluation score.
    """

    data_dir = Path(data_dir)
    frames = []
    for subject_id in range(1, 21):
        file_path = data_dir / f"data{subject_id}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing data file: {file_path}")
        frames.append(pd.read_csv(file_path).to_numpy())

    stacked = np.asarray(frames, dtype=np.float64)
    x = stacked[:, :, 2][:, :, None]
    y_raw = stacked[:, :, 1]
    y = (y_raw - y_raw.mean()) / y_raw.std()
    return [subject_x for subject_x in x], [subject_y for subject_y in y]

