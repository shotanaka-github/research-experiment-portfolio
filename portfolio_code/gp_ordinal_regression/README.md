# Gaussian Process Ordinal Regression

7 段階の順序評定データに対して、GPflow の `Ordinal` likelihood を用いた
Gaussian process ordinal regression を実行する整理版である。

## この整理版で残したもの

- `run_gp_ordinal_regression.py`
  - 各被験者データに対する学習と予測
- `data/`
  - 順序評定 CSV

## 実行例

```bash
cd portfolio_code/gp_ordinal_regression
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_gp_ordinal_regression.py
```

`tensorflow` と `gpflow` を使うため、他の実験より依存関係が重い。
