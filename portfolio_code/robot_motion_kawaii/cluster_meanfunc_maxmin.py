import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import differential_evolution

# ユーザー定義モジュール
import load_psycho_data2_2
from load_psycho_data2_2 import SUBJECT_IDS
from model.KPCA5 import SparseKPCA

# --- 設定 ---
CONFIG = {
    'model_dim': 20,
    'jitter': 1e-6,
    'output_dir': "kawaii_motion_4",  # ★ 保存ディレクトリ
    
    # ★ クラスタ数 (デンドログラムで切る数)
    'num_clusters': 4, 
    
    # 探索設定
    'de_options': {
        'strategy': 'best1bin', 'maxiter': 5000, 'popsize': 50, 'tol': 1e-6,
        'seed': 42, 'disp': False
    },
    'fixed_params': {'length': 2.0, 'noise_level': 1.0}
}

# --- 1. 最適化関数 ---
def find_cluster_optima(model, cluster_indices, bounds):
    """
    指定されたクラスタメンバー(indices)の平均効用関数の最大・最小を探索
    """
    # クラスタ平均の重みベクトルを計算
    # KRRの線形性により、出力の平均 = 平均重みによる出力
    a_cluster_mean = np.mean(model.a[cluster_indices], axis=0)

    # 予測関数 (クラスタ平均)
    def predict_avg(x):
        # x: (16,) -> (1, 16)
        x_in = x.reshape(1, -1)
        Kn = model._rbf(x_in, model.x_all, model.params["length"])
        return np.dot(Kn, a_cluster_mean)[0]

    # 最大化 (符号反転)
    res_max = differential_evolution(lambda x: -predict_avg(x), bounds, **CONFIG['de_options'])
    
    # 最小化
    res_min = differential_evolution(predict_avg, bounds, **CONFIG['de_options'])
    
    return res_max.x, -res_max.fun, res_min.x, res_min.fun

# --- 2. デンドログラム & クラスタリング ---
def perform_clustering_and_save(model, x_dim):
    """デンドログラム作成、クラスタ分け、平均関数の探索・保存"""
    
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Clustering ---")
    
    # 1. 距離行列計算
    if model.a is None: return
    diff = model.a[:, None, :] - model.a[None, :, :]
    dist_sq = np.einsum('ijn,nm,ijm->ij', diff, model.K, diff)
    dist_mat = np.sqrt(np.maximum(dist_sq, 0.0))
    
    # 2. 階層的クラスタリング (Ward法)
    linked = linkage(squareform(dist_mat, checks=False), method="ward")
    
    # 3. デンドログラム保存 (確認用)
    plt.figure(figsize=(12, 8))
    dendrogram(linked, labels=SUBJECT_IDS, color_threshold=linked[-(CONFIG['num_clusters']-1), 2])
    plt.title(f"Dendrogram (Cut into {CONFIG['num_clusters']} clusters)")
    plt.xlabel("Subject ID")
    plt.ylabel("Distance")
    plt.savefig(os.path.join(output_dir, "dendrogram.pdf"))
    plt.close()
    print(f"デンドログラムを保存しました: {output_dir}/dendrogram.pdf")

    # 4. クラスタIDの割り当て
    # criterion='maxclust' で指定したクラスタ数に分割
    cluster_labels = fcluster(linked, t=CONFIG['num_clusters'], criterion='maxclust')
    
    # メンバー表の保存
    df_members = pd.DataFrame({'Subject_ID': SUBJECT_IDS, 'Cluster_ID': cluster_labels})
    df_members.to_csv(os.path.join(output_dir, "cluster_members.csv"), index=False)
    
    # 5. クラスタごとの処理
    bounds = [(-1.0, 1.0)] * x_dim
    print(f"\n--- Cluster Average Optimization (Total {CONFIG['num_clusters']} clusters) ---")
    
    # クラスタIDは 1 から始まる
    for c_id in range(1, CONFIG['num_clusters'] + 1):
        # メンバーのインデックス抽出
        indices = np.where(cluster_labels == c_id)[0]
        count = len(indices)
        
        print(f"Cluster {c_id}: {count} members ... ", end="", flush=True)
        
        if count == 0:
            print("Skipped (Empty)")
            continue
            
        # 平均関数の最大・最小探索
        mx, mv, mn, mnv = find_cluster_optima(model, indices, bounds)
        
        # --- 保存 ---
        # kawaii_motion_cluster_{ID}_1 (Max)
        f_max = os.path.join(output_dir, f"kawaii_motion_cluster{c_id}_1.csv")
        pd.DataFrame(mx.reshape(1, -1)).to_csv(f_max, header=False, index=False)
        
        # kawaii_motion_cluster_{ID}_2 (Min)
        f_min = os.path.join(output_dir, f"kawaii_motion_cluster{c_id}_2.csv")
        pd.DataFrame(mn.reshape(1, -1)).to_csv(f_min, header=False, index=False)
        
        print(f"Done (Max:{mv:.3f}, Min:{mnv:.3f})")

    print(f"\n完了しました。'{output_dir}' を確認してください。")

# --- 3. メイン ---
def main():
    print("データ読み込み & 学習中...")
    x_train, y_train = load_psycho_data2_2.load_data()
    n_dims = x_train[0].shape[1]

    model = SparseKPCA(
        x_list=x_train, y_list=y_train, params=CONFIG['fixed_params'], 
        modelDim=CONFIG['model_dim'], jitter=CONFIG['jitter']
    )
    model.fit()
    
    if model.Z is None or np.all(model.Z == 0):
        print("エラー: 学習失敗")
        return

    # クラスタ分析 & 保存
    perform_clustering_and_save(model, n_dims)

if __name__ == '__main__':
    main()