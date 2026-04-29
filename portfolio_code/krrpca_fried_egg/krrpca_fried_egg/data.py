from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_fried_egg_data(
    feature_dir: str | Path,
    label_dir: str | Path,
    num_people: int = 40,
    label_column: str = "Label",
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Load and standardize the fried-egg feature and preference data."""

    feature_dir = Path(feature_dir)
    label_dir = Path(label_dir)

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    subject_ids = np.arange(1, num_people + 1)

    for subject_id in subject_ids:
        feature_path = feature_dir / f"person_{subject_id}_data.csv"
        label_path = label_dir / f"data{subject_id}_pca_5d.csv"

        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        x_frame = pd.read_csv(feature_path)
        y_frame = pd.read_csv(label_path)
        if label_column not in y_frame.columns:
            raise KeyError(f"Column '{label_column}' is not available in {label_path}")

        x_values = x_frame.to_numpy(dtype=np.float64)
        y_values = y_frame[label_column].to_numpy(dtype=np.float64)
        if len(x_values) != len(y_values):
            raise ValueError(
                f"Feature/label length mismatch for subject {subject_id}: "
                f"{len(x_values)} features vs {len(y_values)} labels"
            )

        x_list.append(x_values)
        y_list.append(y_values)

    x_all = np.concatenate(x_list, axis=0)
    x_mean = x_all.mean(axis=0)
    x_std = x_all.std(axis=0)
    x_std = np.where(x_std == 0.0, 1.0, x_std)
    x_list = [(x - x_mean) / x_std for x in x_list]

    standardized_y_list: list[np.ndarray] = []
    for y_values in y_list:
        y_mean = y_values.mean()
        y_std = y_values.std()
        if y_std == 0.0:
            standardized_y_list.append(y_values - y_mean)
        else:
            standardized_y_list.append((y_values - y_mean) / y_std)
    y_list = standardized_y_list

    return x_list, y_list, subject_ids
