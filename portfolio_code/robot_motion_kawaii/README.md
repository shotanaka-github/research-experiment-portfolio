# Robot Motion Preference Experiment

ロボット動作パラメータに対する主観評価データを、共通基底型の kernel ridge regression + kernel PCA で解析し、
さらにクラスタごとの平均効用関数を抽出する。

## Source

- `main_sota.py`: KRR + KPCA の学習、潜在空間可視化、CCA を含む性格特性との関連解析
- `tune_hyperparameters.py`: `length` と `noise_level` の CV 探索
- `tune_hyperparameters_GPU.py`: GPU 上で同じ CV 探索を回す PyTorch 実装
- `cluster_meanfunc_maxmin.py`: クラスタ平均効用関数の代表点抽出
- `load_psycho_data2_2.py`: 被験者ごとの 16 次元動作パラメータと応答の読み込み
- `model/KPCA5.py`: 共通基底型 KRR + KPCA モデル
- `data/`: 入力パラメータと応答 CSV
- `bigfive_scores.csv`: 性格尺度との対応づけに使う表

## Tree

```text
robot_motion_kawaii/
├── README.md
├── requirements.txt
├── requirements_gpu.txt
├── bigfive_scores.csv
├── load_psycho_data2_2.py
├── main_sota.py
├── tune_hyperparameters.py
├── tune_hyperparameters_GPU.py
├── cluster_meanfunc_maxmin.py
├── model/
│   └── KPCA5.py
└── data/
```

## Run

```bash
cd portfolio_code
python3 -m venv robot_motion_kawaii/.venv
source robot_motion_kawaii/.venv/bin/activate
cd robot_motion_kawaii
pip install -r requirements.txt
python tune_hyperparameters.py
python main_sota.py --cv-results hyperparameter_search/cv_results.csv
python ../run_robot_motion_kawaii_clustering.py --cv-results robot_motion_kawaii/hyperparameter_search/cv_results.csv
```
