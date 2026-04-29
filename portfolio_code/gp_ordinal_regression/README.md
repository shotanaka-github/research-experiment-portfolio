# Gaussian Process Ordinal Regression

7 段階の順序評定データに対して、GPflow の `Ordinal` likelihood を用いた
Gaussian process ordinal regression を当て、その近似事後を GPPCA で要約する。

## Source

- `run_gp_ordinal_regression.py`: ordinal GP の学習、shared basis への写像、GPPCA による低次元化
- `data/`: 順序評定 CSV

## Tree

```text
gp_ordinal_regression/
├── README.md
├── requirements.txt
├── run_gp_ordinal_regression.py
└── data/
```

## Run

```bash
cd portfolio_code/gp_ordinal_regression
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_gp_ordinal_regression.py --basis-size 64 --epochs 200
```

shared basis 上の ordinal GP 事後は、`predict_f` の平均と marginal variance による対角共分散近似として GPPCA に渡している。

`tensorflow` と `gpflow` を使うため、他の実験より依存関係が重い。
