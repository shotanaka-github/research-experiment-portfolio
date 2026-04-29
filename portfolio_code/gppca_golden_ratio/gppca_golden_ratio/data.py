from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_golden_ratio_data(data_dir: str | Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load golden-ratio ratings as subject-wise `(x, y)` lists."""

    data_dir = Path(data_dir)
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for subject_id in range(1, 22):
        file_path = data_dir / f"data{subject_id}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing data file: {file_path}")

        frame = pd.read_csv(file_path)
        x_subject = frame["aspect_ratio"].to_numpy(dtype=np.float64).reshape(-1, 1)
        y_raw = frame["beauty"].to_numpy(dtype=np.float64)
        y_std = float(y_raw.std())
        if y_std <= 1e-12:
            y_subject = y_raw - y_raw.mean()
        else:
            y_subject = (y_raw - y_raw.mean()) / y_std

        x_list.append(x_subject)
        y_list.append(y_subject)

    return x_list, y_list
