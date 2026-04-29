# Research Experiment Portfolio

このリポジトリは、感性評価・選好学習まわりの研究コードを GitHub で共有しやすい形に整理したものである。

## 環境構築

依存関係は実験ごとに異なるため、`portfolio_code/` 配下でプロジェクトごとに仮想環境を作る形にしている。

```bash
cd portfolio_code
chmod +x setup_env.sh
./setup_env.sh krrpca_fried_egg
```

詳しくは [portfolio_code/ENVIRONMENT.md](/Users/komorilab/Documents/New%20project/portfolio_code/ENVIRONMENT.md) を参照。

## 含めた主な実験

- `portfolio_code/krrpca_fried_egg`
  - 目玉焼き特徴量に対する kernel ridge regression + kernel PCA
- `portfolio_code/shape_preference_rectangle`
  - 1 次元形状パラメータに対する KRR + KPCA
- `portfolio_code/robot_motion_kawaii`
  - ロボット動作パラメータに対する KRR + KPCA
- `portfolio_code/kernel_ordinal_regression`
  - probit 型のカーネル順序回帰
- `portfolio_code/gp_ordinal_regression`
  - ordinal GP の近似事後を GPPCA で要約する順序回帰版
- `portfolio_code/gppca_golden_ratio`
  - 黄金比評価データに対する GPPCA
- `portfolio_code/pairwise_gppca`
  - 実 pairwise データと誘導 pairwise データを切り替えて動かせる GPPCA

## メモ

- `portfolio_code/method_inventory.md` に、KRR 関連の公開候補を一覧化した
