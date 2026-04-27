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
pip install -r requirements.txt
python run_gp_ordinal_regression.py
```
