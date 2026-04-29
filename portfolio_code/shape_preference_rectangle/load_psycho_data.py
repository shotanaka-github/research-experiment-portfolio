from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

"""
def load_data():
    data = []
    for i in range(1, 41):#41=40人
        df = pd.read_csv(f'reduction_results_per_file/pca/data{i}_pca.csv')
        print(f"data{i}を読み込んでいます。")
        data.append(df.values)
    data = np.array(data)
    #print('data shape:', data.shape)
    X = data[:, :, 2][:, :, None]
    # X = data[:, :, 0][:, :, None]
    y = (data[:, :, 1] - data[:, :, 1].mean())/data[:, :, 1].std()
    return X, y
"""
"""
def load_data():
    data = []
    for i in range(1, 21):
        df = pd.read_csv(f'data/data{i}.csv')
        print(f"data{i}を読み込んでいます。")
        data.append(df.values)
    data = np.array(data)
    X = data[:, :, 2][:, :, None]
    # X = data[:, :, 0][:, :, None]
    y = (data[:, :, 1] - data[:, :, 1].mean())/data[:, :, 1].std()
    return X, y
"""
import numpy as np
import pandas as pd

def load_data(data_dir: str | Path = "data", num_people: int = 21):
    """
    【修正版】
    各被験者のデータを読み込み、Xとyのリストとして返す。
    yは「被験者ごと」にzスコア化（標準化）する。
    """
    data_dir = Path(data_dir)
    x_list = []
    y_list = []

    for i in range(1, num_people + 1):
        df = pd.read_csv(data_dir / f"data{i}.csv")
        print(f"data{i}.csv を読み込んでいます。 (shape: {df.shape})")
        
        # 観測点 (X) と 効用値 (y) を抽出
        # X: 3列目 (index 2), y: 2列目 (index 1)
        x_subject = df.values[:, 2].reshape(-1, 1)
        y_subject_raw = df.values[:, 1]
        
        # yを被験者ごとにzスコア化（標準化）
        y_subject_std = np.std(y_subject_raw)
        if y_subject_std == 0:
            y_subject_z = y_subject_raw * 0.0
        else:
            y_subject_z = (y_subject_raw - np.mean(y_subject_raw)) / y_subject_std
        
        x_list.append(x_subject)
        # yも (N, 1) の形状にしてリストに追加
        y_list.append(y_subject_z.reshape(-1, 1)) 
            
    print(f"\n合計 {len(x_list)} 人分のデータを読み込みました。")
    return x_list, y_list

