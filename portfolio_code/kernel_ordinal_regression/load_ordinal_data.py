from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_ordinal_data(data_dir: Path, num_people: int = 20) -> tuple[np.ndarray, np.ndarray]:
    x_list = []
    y_list = []
    for index in range(1, num_people + 1):
        frame = pd.read_csv(data_dir / f"data{index}.csv")
        x_list.append(frame.iloc[:, 2].to_numpy(dtype=np.float64).reshape(-1, 1))
        y_list.append(frame.iloc[:, 1].to_numpy(dtype=np.int64))
    return np.array(x_list), np.array(y_list)
