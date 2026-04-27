# Color Preference Experiment

Lab 色空間上の単色刺激に対して、参照色に近いと感じる色を二肢選択で収集する実験コードである。

## この整理版で残したもの

- `2alternative.py`
  - PsychoPy 実験本体
- `2alternative.psyexp`
  - Builder 形式の実験定義
- `ego/stimulation/color.py`
  - 単色刺激画像の生成と候補点選択
- `postprocess/main.py`
  - 収集済みデータから Pairwise GP で選好面を復元する後処理
- `Lab_color_space.png`
  - Lab 空間の説明用画像
- `ref.jpg`
  - 基準刺激画像

## 省いたもの

- 実験参加者名を含む生データは、この公開版には含めていない
- 生データを置く場合は `data/` を作り、`*_result.pt` と `*_response.pt` を配置する
- `postprocess/name_list*.csv` は公開版ではテンプレートに差し替えてある

## 実行例

```bash
cd portfolio_code/color_preference_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python 2alternative.py
```

後処理:

```bash
python postprocess/main.py --data-dir data --output-dir postprocess/result
```
