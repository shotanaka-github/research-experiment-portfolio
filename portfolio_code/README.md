# Research Code Portfolio

応募用に見せやすいよう、研究コードを整理してまとめた。

## Environment

各フォルダには個別の `requirements.txt` を置いている。
まとめて環境を作る代わりに、必要な実験だけ個別に仮想環境を作る運用にしている。

```bash
cd portfolio_code
./setup_env.sh gppca_golden_ratio
```

補足は [ENVIRONMENT.md](/Users/komorilab/Documents/New%20project/portfolio_code/ENVIRONMENT.md) にまとめた。

## Projects

- `gppca_golden_ratio`: 黄金比の主観評価データに対する Gaussian-process exponential-family PCA
- `pairwise_gppca`: 実 pairwise データと誘導 pairwise データを切り替えて動かせる GPPCA
- `krrpca_fried_egg`: 目玉焼きデータに対する kernel-ridge-regression PCA
- `color_preference_lab`: Lab 色空間での二肢選択色選好実験
- `shape_preference_rectangle`: 1 次元形状パラメータに対する KRR + KPCA 実験
- `robot_motion_kawaii`: ロボット動作パラメータに対する KRR + KPCA 実験
- `kernel_ordinal_regression`: probit 型のカーネル順序回帰
- `gp_ordinal_regression`: Gaussian process 順序回帰

元コードは別フォルダに残したまま、この `portfolio_code/` 配下に整理版を新しく作っている。
探索途中の派生スクリプトや重複結果を減らし、公開して説明しやすい最小構成へ寄せた。
