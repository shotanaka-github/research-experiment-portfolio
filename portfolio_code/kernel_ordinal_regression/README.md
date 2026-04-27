# Kernel Ordinal Regression

7 段階の順序評定データに対して、probit 型のカーネル順序回帰を各被験者ごとに学習し、
得られた係数ベクトルを PCA 的に要約する整理版である。

## この整理版で残したもの

- `model/KROR_PCA2.py`
  - JAX による probit 型カーネル順序回帰 + PCA
- `load_ordinal_data.py`
  - 被験者ごとの入力と順序ラベルの読み込み
- `run_kernel_ordinal_regression.py`
  - 学習、潜在座標保存、予測曲線保存
- `data/`
  - 20 名分の順序評定 CSV

## 実行例

```bash
pip install -r requirements.txt
python run_kernel_ordinal_regression.py
```
