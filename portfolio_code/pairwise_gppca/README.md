# Pairwise GPPCA

複数の選好データセットに対して、pairwise preference learning の近似事後分布を
Gaussian-process exponential-family PCA (GPPCA) で要約する。

## Datasets

- `thurstone_pairwise`
  - 実際に二肢選択として収集した 6 次元刺激の pairwise データ
  - 既定の実行例
- `golden_ratio_induced`
  - 黄金比の 7 段階評定から誘導した pairwise comparison
  - 元データは順序評定であり、pairwise は二次的に生成したもの

## Source

- `pairwise_gppca/model.py`: 共通の Pairwise GPPCA 本体
- `pairwise_gppca/types.py`: dataset adapter が返す共通 dataclass
- `pairwise_gppca/datasets/`: dataset ごとの loader
- `scripts/run_pairwise_gppca.py`: dataset 切替つき CLI
- `tests/test_pairwise_gppca.py`: golden-ratio induced と Thurstone 用 smoke test
- `data/golden_ratio/`: 順序評定データ
- `data/thurstone/`: 実 pairwise データ

## Tree

```text
pairwise_gppca/
├── README.md
├── requirements.txt
├── data/
│   ├── golden_ratio/
│   └── thurstone/
├── pairwise_gppca/
│   ├── data.py
│   ├── model.py
│   ├── types.py
│   └── datasets/
├── scripts/
│   └── run_pairwise_gppca.py
└── tests/
    └── test_pairwise_gppca.py
```

## Notes

- pairwise likelihood は probit `P(i > j) = Phi((f_i - f_j) / sigma)` を使う
- Thurstone のような実 pairwise データでは、初期化に win-loss 差分の標準化スコアを使う
- golden ratio では元の評定値を初期化に使い、pairwise 制約は評定差から生成する
- 1 次元データセットでは予測曲線の CSV / PDF を出力する
- 多次元データセットでは共通 basis 上の posterior mean と latent 座標を保存する

## Run

```bash
cd portfolio_code/pairwise_gppca
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pairwise_gppca.py --dataset thurstone_pairwise --basis-size 64 --epochs 40 --output-dir output/thurstone
python scripts/run_pairwise_gppca.py --dataset golden_ratio_induced --basis-size 32 --epochs 40 --output-dir output/golden_ratio
python -m unittest tests.test_pairwise_gppca
```
