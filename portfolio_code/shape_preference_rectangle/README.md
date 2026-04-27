# Shape Preference Rectangle Experiment

1 次元の形状パラメータを入力とする選好データに対して、共通基底型の kernel ridge regression + kernel PCA を適用した整理版である。

## この整理版で残したもの

- `rectangle.py`
  - ハイパーパラメータ探索と最終学習
- `load_psycho_data.py`
  - 被験者ごとの入力と応答の読み込み
- `model/KPCA5.py`
  - 共通基底型 KRR + KPCA モデル
- `data/`
  - 被験者ごとの観測 CSV

## 実行例

```bash
pip install -r requirements.txt
python rectangle.py
```
