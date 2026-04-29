# Research Code Portfolio

研究コードを GitHub で共有しやすい形に整理してまとめた。

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
- `shape_preference_rectangle`: 1 次元形状パラメータに対する KRR + KPCA 実験
- `robot_motion_kawaii`: ロボット動作パラメータに対する KRR + KPCA 実験
- `kernel_ordinal_regression`: probit 型のカーネル順序回帰
- `gp_ordinal_regression`: ordinal GP の近似事後を GPPCA で要約する順序回帰版
