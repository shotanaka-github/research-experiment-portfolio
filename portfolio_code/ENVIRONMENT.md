# Environment Setup

各サブプロジェクトは依存関係がかなり異なるため、共通の単一環境ではなく
「プロジェクトごとの仮想環境」を作る前提にしている。

## 最短手順

```bash
cd portfolio_code
chmod +x setup_env.sh
./setup_env.sh krrpca_fried_egg
```

これで `portfolio_code/krrpca_fried_egg/.venv` が作られ、対応する `requirements.txt` が入る。

## 例

```bash
cd portfolio_code
./setup_env.sh gppca_golden_ratio
./setup_env.sh kernel_ordinal_regression
```

## 補足

- `gp_ordinal_regression` は `tensorflow` / `gpflow` を使うため最も重い
- 軽めに試すなら `krrpca_fried_egg` か `gppca_golden_ratio` から始めるのが無難
