from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUBJECT_IDS = [
    101,102,103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
    201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 216, 301
]

def load_data(data_dir: str | Path = "data"):
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parent / data_dir
    X_data = [] 
    y_data = [] 
    
    for subject_id in SUBJECT_IDS: 
        x_path = data_dir / f"parameters_{subject_id}_result.csv"
        x_df = pd.read_csv(x_path, header=None)
        X_data.append(x_df.values)

        y_path = data_dir / f"parameters_{subject_id}_response.csv"
        y_df = pd.read_csv(y_path, header=None)
        y_values = y_df.values.ravel() 

        y_mean = y_values.mean()
        y_std = y_values.std()
        
        if y_std == 0:
            y_standardized = y_values - y_mean
        else:
            y_standardized = (y_values - y_mean) / y_std
        
        y_data.append(y_standardized)

    X = np.array(X_data)
    y = np.array(y_data)
    return X, y
