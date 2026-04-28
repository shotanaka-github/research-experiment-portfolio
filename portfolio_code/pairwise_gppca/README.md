# Pairwise GPPCA

複数の選好データセットに対して、pairwise preference learning の近似事後分布を
Gaussian-process exponential-family PCA (GPPCA) で要約する整理版である。

このエントリでは、同じ `PairwiseGPPCA` 実装に対して複数の dataset adapter を差し替えられる。

## 含めたデータセット

- `thurstone_pairwise`
  - 実際に二肢選択として収集した 6 次元刺激の pairwise データ
  - 既定の実行例
- `golden_ratio_induced`
  - 黄金比の 7 段階評定から誘導した pairwise comparison
  - 元データは順序評定であり、pairwise は二次的に生成したもの
- `color_lab_pairwise`
  - Lab 色空間で収集した二肢選択データ
  - adapter は含めるが、生の `*_result.pt` / `*_response.pt` は公開版には含めていない

## 構成

- `pairwise_gppca/model.py`
  - 共通の Pairwise GPPCA 本体
- `pairwise_gppca/types.py`
  - dataset adapter が返す共通 dataclass
- `pairwise_gppca/datasets/`
  - dataset ごとの loader
- `scripts/run_pairwise_gppca.py`
  - dataset 切替つき CLI
- `tests/test_pairwise_gppca.py`
  - golden-ratio induced と Thurstone 用 smoke test
- `data/golden_ratio/`
  - 順序評定データ
- `data/thurstone/data10/`
  - 実 pairwise データ

## 実装メモ

- pairwise likelihood は probit `P(i > j) = Phi((f_i - f_j) / sigma)` を使う
- Thurstone / color のような実 pairwise データでは、初期化に win-loss 差分の標準化スコアを使う
- golden ratio では元の評定値を初期化に使い、pairwise 制約は評定差から生成する
- 1 次元データセットでは予測曲線の CSV / PDF を出力する
- 多次元データセットでは共通 basis 上の posterior mean と latent 座標を保存する

## 実行例

```bash
cd portfolio_code/pairwise_gppca
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pairwise_gppca.py --dataset thurstone_pairwise --basis-size 64 --epochs 40 --output-dir output/thurstone
python scripts/run_pairwise_gppca.py --dataset golden_ratio_induced --basis-size 32 --epochs 40 --output-dir output/golden_ratio
python -m unittest tests.test_pairwise_gppca
```

`color_lab_pairwise` を使う場合は、生データを `portfolio_code/color_preference_lab/data/` に置くか、
`--data-dir` と `--name-list-file` を明示する。
