# Kernel Ridge Regression Related Method Inventory

このワークスペースで確認できた、kernel ridge regression 関連の公開候補は次の通り。

## すでに整理済み

- `krrpca_fried_egg`
  - 共通基底上の kernel ridge regression + kernel PCA

## 今回追加した公開候補

- `krrpca_fried_egg/scripts/run_krrpca_clustering.py`
  - RKHS 距離に基づく Ward 法クラスタリング版
- `shape_preference_rectangle`
  - 1 次元形状パラメータの KRR + KPCA 実験
- `robot_motion_kawaii`
  - ロボット動作パラメータの KRR + KPCA 実験
- `color_preference_lab`
  - 色選好の収集実験と Pairwise GP 後処理

## この整理版に含めなかったもの

- 旧フォルダに散在する大量の試行錯誤スクリプト
- 個人名を含む色実験の生データ
- 生成済み PDF を大量に含む中間結果一式
