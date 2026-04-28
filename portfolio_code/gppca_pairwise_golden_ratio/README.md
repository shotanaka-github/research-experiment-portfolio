# Pairwise GPPCA For Golden Ratio Preference Data

黄金比に関する 7 段階評定データを、二肢選択の比較データへ変換して扱う
Gaussian-process exponential-family PCA (GPPCA) の整理版である。

各被験者について:

1. 評定値の大小から pairwise comparison `(winner, loser)` を生成する
2. GP preference learning を Laplace 近似で学習し、潜在効用関数のガウス近似事後分布を得る
3. その事後分布を GPPCA で低次元潜在空間へ要約する

## この整理版で残したもの

- `gppca_pairwise_golden_ratio/data.py`
  - CSV 読み込みと pairwise comparison 生成
- `gppca_pairwise_golden_ratio/model.py`
  - GP preference posterior + GPPCA
- `scripts/run_gppca_pairwise.py`
  - 学習、潜在座標保存、比較数保存、予測曲線出力
- `data/`
  - 元データ
- `tests/test_pairwise_gppca.py`
  - 最小限の smoke test

## 実装メモ

- 入力には `aspect_ratio` 列を使う
- 評価が同点のペアは比較データに入れない
- pairwise likelihood は probit `P(i > j) = Phi((f_i - f_j) / sigma)` を使う
- `GPflow` には ordinal likelihood があるが、本整理版は外部依存を重くせずに読めるよう NumPy の自前実装に寄せている
- `BoTorch` には `PairwiseGP` があるが、本コードは GPPCA へつなぐ都合上、各被験者の近似事後分布を明示的に保持する形で整理している

## 実行例

```bash
cd portfolio_code/gppca_pairwise_golden_ratio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_gppca_pairwise.py --epochs 120 --basis-size 32 --output-dir output
python -m unittest tests.test_pairwise_gppca
```

出力先には、学習済みモデル、潜在座標、各被験者の比較数、元の GP posterior と
GPPCA 再構成 posterior の効用曲線が保存される。`matplotlib` がない環境では図保存のみ自動でスキップする。
