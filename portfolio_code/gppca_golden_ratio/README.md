# GPPCA For Golden Ratio Preference Data

黄金比に関する主観評価データを対象に、各被験者の評価関数を Gaussian process で表現し、
その事後分布を指数型分布族上で低次元化するコードである。

## Source

- `gppca_golden_ratio/data.py`: 21 名分の CSV を読み込み、`aspect_ratio` と評価値を整形する
- `gppca_golden_ratio/model.py`: GPPCA の中核クラス
- `scripts/run_gppca.py`: 学習、予測、潜在座標の保存、可視化
- `data/`: 元データ
## Tree

```text
gppca_golden_ratio/
├── README.md
├── requirements.txt
├── data/
├── gppca_golden_ratio/
│   ├── data.py
│   └── model.py
└── scripts/
    └── run_gppca.py
```

## Run

```bash
cd portfolio_code/gppca_golden_ratio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_gppca.py --epochs 200 --basis-size 64 --output-dir output
```

出力先には、学習済みモデル、潜在座標、被験者ごとの推定曲線が保存される。
`matplotlib` が入っていない環境では、図の保存だけ自動でスキップする。
