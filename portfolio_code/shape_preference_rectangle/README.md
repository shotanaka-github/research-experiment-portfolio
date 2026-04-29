# Shape Preference Rectangle Experiment

1 次元の形状パラメータを入力とする選好データに対して、共通基底型の kernel ridge regression + kernel PCA を適用する。

## Source

- `rectangle.py`: ハイパーパラメータ探索と最終学習
- `load_psycho_data.py`: 被験者ごとの入力と応答の読み込み
- `model/KPCA5.py`: 共通基底型 KRR + KPCA モデル
- `data/`: 被験者ごとの観測 CSV
## Tree

```text
shape_preference_rectangle/
├── README.md
├── requirements.txt
├── load_psycho_data.py
├── rectangle.py
├── model/
│   └── KPCA5.py
└── data/
```

## Run

```bash
cd portfolio_code/shape_preference_rectangle
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
python run_shape_preference_rectangle.py
```
