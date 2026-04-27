# KRRPCA For Fried-Egg Preference Data

目玉焼き画像から抽出した特徴量に対して、各被験者の評価関数を
kernel ridge regression で推定し、その重みを kernel PCA で低次元化するコードである。

## 元コードから整理した点

- 被験者ごとの観測点が異なる前提を明示した
- 共通基底を `x_all` として定義し、各被験者の重みをその基底上で推定する形に統一した
- ハイパーパラメータ探索を、被験者ごとに独立に分割できる cross validation に整理した
- 実験途中の派生スクリプトや重複モデルを削って、再実行に必要な最小構成へ絞った
- `matplotlib` がない環境でも、CSV と学習済みモデルの出力までは動くようにした

## この整理版で残したもの

- `krrpca_fried_egg/data.py`
  - 特徴量 CSV とラベル CSV の読み込み
- `krrpca_fried_egg/model.py`
  - 共通基底上の kernel ridge regression + kernel PCA
- `scripts/search_hyperparams.py`
  - ハイパーパラメータ探索
- `scripts/run_krrpca.py`
  - 最終学習、潜在空間・寄与率の保存
- `scripts/run_krrpca_clustering.py`
  - RKHS 距離に基づく Ward 法クラスタリングとクラスタ要約
- `data/features`
  - 被験者ごとの特徴量
- `data/labels`
  - 被験者ごとのラベル

## 実行例

```bash
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
  --length 0.3 \
  --noise-level 0.1 \
  --basis-size 256 \
  --output-dir output/final
```

```bash
python scripts/run_krrpca_clustering.py \
  --length 0.3 \
  --noise-level 0.1 \
  --basis-size 256 \
  --num-clusters 4 \
  --output-dir output/clustering
```

クラスタリング版では、各被験者の推定重みベクトル間の RKHS 距離を用いて
Ward 法の階層クラスタリングを行い、クラスタ割当・距離行列・代表被験者を保存する。
