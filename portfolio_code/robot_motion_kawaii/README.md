# Robot Motion Preference Experiment

ロボット動作パラメータに対する主観評価データを、共通基底型の kernel ridge regression + kernel PCA で解析し、
さらにクラスタごとの平均効用関数を抽出するための整理版である。

## この整理版で残したもの

- `main_sota.py`
  - 主解析スクリプト
- `cluster_meanfunc_maxmin.py`
  - クラスタ平均効用の最大点・最小点探索
- `load_psycho_data2_2.py`
  - 被験者ごとの 16 次元動作パラメータと応答の読み込み
- `model/KPCA5.py`
  - 共通基底型 KRR + KPCA モデル
- `data/`
  - 入力パラメータと応答 CSV
- `bigfive_scores.csv`
  - 性格尺度との対応づけに使う表

## 実行例

```bash
cd portfolio_code
python3 -m venv robot_motion_kawaii/.venv
source robot_motion_kawaii/.venv/bin/activate
cd robot_motion_kawaii
pip install -r requirements.txt
python main_sota.py
python ../run_robot_motion_kawaii_clustering.py
```
