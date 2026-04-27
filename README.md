# Research Experiment Portfolio

このリポジトリは、感性評価・選好学習まわりの研究コードを GitHub で共有しやすい形に整理したものである。

## 含めた主な実験

- `portfolio_code/krrpca_fried_egg`
  - 目玉焼き特徴量に対する kernel ridge regression + kernel PCA
- `portfolio_code/shape_preference_rectangle`
  - 1 次元形状パラメータに対する KRR + KPCA
- `portfolio_code/robot_motion_kawaii`
  - ロボット動作パラメータに対する KRR + KPCA
- `portfolio_code/color_preference_lab`
  - Lab 色空間での二肢選択色選好実験
- `portfolio_code/kernel_ordinal_regression`
  - probit 型のカーネル順序回帰
- `portfolio_code/gp_ordinal_regression`
  - Gaussian process 順序回帰
- `portfolio_code/gppca_golden_ratio`
  - 黄金比評価データに対する GPPCA

## メモ

- `portfolio_code/method_inventory.md` に、KRR 関連の公開候補を一覧化した
- 色実験の生データは個人名を含むため、この公開版からは除外した
- 元の探索用フォルダは別のローカルディレクトリに保持したまま、ここでは公開用の整理版だけを置いている
