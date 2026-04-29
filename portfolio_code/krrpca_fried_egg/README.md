# KRRPCA For Fried-Egg Preference Data

目玉焼き画像から抽出した特徴量に対して、各被験者の評価関数を
kernel ridge regression で推定し、その重みを kernel PCA で低次元化するコードである。

## Source

- `krrpca_fried_egg/data.py`: 特徴量 CSV とラベル CSV の読み込み
- `krrpca_fried_egg/model.py`: 共通基底上の kernel ridge regression + kernel PCA
- `scripts/search_hyperparams.py`: ハイパーパラメータ探索
- `scripts/run_krrpca.py`: 最終学習、潜在座標・寄与率の保存
- `scripts/run_krrpca_clustering.py`: RKHS 距離に基づく Ward 法クラスタリング
- `data/features/`: 被験者ごとの特徴量
- `data/labels/`: 被験者ごとのラベル

## Tree

```text
krrpca_fried_egg/
├── README.md
├── requirements.txt
├── data/
│   ├── features/
│   └── labels/
├── krrpca_fried_egg/
│   ├── data.py
│   └── model.py
└── scripts/
    ├── search_hyperparams.py
    ├── run_krrpca.py
    └── run_krrpca_clustering.py
```

## Run

```bash
cd portfolio_code/krrpca_fried_egg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/search_hyperparams.py \
  --length-min 0.1 \
  --length-max 1.0 \
  --length-points 4 \
  --noise-min 0.01 \
  --noise-max 0.5 \
  --noise-points 4 \
  --basis-size 256 \
  --output-dir output/hyperparams
```

```bash
python scripts/run_krrpca.py \
  --best-params-file output/hyperparams/best_params.json \
  --basis-size 256 \
  --output-dir output/final
```

```bash
python scripts/run_krrpca_clustering.py \
  --best-params-file output/hyperparams/best_params.json \
  --basis-size 256 \
  --num-clusters 4 \
  --output-dir output/clustering
```

クラスタリング版では、各被験者の推定重みベクトル間の RKHS 距離を用いて
Ward 法の階層クラスタリングを行い、クラスタ割当・距離行列・代表被験者を保存する。
