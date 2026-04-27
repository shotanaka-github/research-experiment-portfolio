import numpy as np
import pandas as pd
import os
import joblib
from itertools import product, combinations
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from matplotlib.backends.backend_pdf import PdfPages 

import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# ★ SUBJECT_IDS をインポート
import load_psycho_data2_2
from load_psycho_data2_2 import SUBJECT_IDS 
from model.KPCA5 import SparseKPCA

# --- 1. 補助関数 ---

def zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    sd = np.std(v)
    if sd == 0:
        return v * 0.0
    return (v - np.mean(v)) / sd

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))

# --- 2. 可視化ヘルパー関数 (2D等高線プロット用) ---

def plot_utility_contour(ax, model, x_mean, dim_pair, config, s_idx=None, z_vec=None):
    """
    指定されたaxに2D等高線プロットを描画する内部関数。
    """
    font_sizes = config['font_sizes']
    plot_dim = config['plot_dim'] 
    grid_points = config['grid_points']
    
    # 描画グリッドの作成
    plot_range = np.linspace(-plot_dim, plot_dim, grid_points)
    X_grid, Y_grid = np.meshgrid(plot_range, plot_range)
    
    # (grid_points*grid_points, 16) の入力データを作成
    x_range_for_plot = np.tile(x_mean, (grid_points**2, 1))
    x_range_for_plot[:, dim_pair[0]] = X_grid.ravel()
    x_range_for_plot[:, dim_pair[1]] = Y_grid.ravel()

    # 予測
    if z_vec is not None:
        Z_grid = model.generate_pos(x_range_for_plot, z_vec)[0, :]
    elif s_idx is not None:
        Z_grid = model.predict_subject(x_range_for_plot, s_idx)
    else:
        # 平均効用
        a_avg = model.W0[None, :] 
        Kn = model._rbf(x_range_for_plot, model.x_all, model.params["length"])
        Z_grid = np.einsum("nk,ik->in", Kn, a_avg)[0, :]

    Z_grid = Z_grid.reshape(grid_points, grid_points)

    # 等高線描画
    ax.contourf(X_grid, Y_grid, Z_grid, levels=15, cmap='viridis')
    cf = ax.contour(X_grid, Y_grid, Z_grid, levels=15, colors='black', linewidths=0.5)
    ax.clabel(cf, inline=True, fontsize=font_sizes['ticks'] - 4)

    # 最大・最小点の探索と描画
    max_idx = np.unravel_index(np.argmax(Z_grid), Z_grid.shape)
    min_idx = np.unravel_index(np.argmin(Z_grid), Z_grid.shape)
    
    ax.scatter(X_grid[max_idx], Y_grid[max_idx], marker='*', color='red', s=150, label='Max', zorder=10)
    ax.scatter(X_grid[min_idx], Y_grid[min_idx], marker='x', color='cyan', s=100, label='Min', zorder=10)

    ax.set_xlabel(f"Dim {dim_pair[0]}", fontsize=font_sizes['label'])
    ax.set_ylabel(f"Dim {dim_pair[1]}", fontsize=font_sizes['label'])
    ax.tick_params(axis='both', which='major', labelsize=font_sizes['ticks'])
    ax.set_aspect('equal')

def plot_individual_utility_contours(model: SparseKPCA, config: dict):
    """個人の予測効用関数（等高線）を個別PDFで保存する (ID名使用)"""
    
    output_dir_individual = os.path.join(config['output_dir'], "individual_contours")
    os.makedirs(output_dir_individual, exist_ok=True)
    print(f"\n個人の効用関数（等高線）を '{output_dir_individual}' フォルダに個別保存します...")
    
    x_mean = config['x_mean']
    n_dims = x_mean.shape[0] # 16次元

    # 120通りのペアでループ
    for dim_pair in combinations(range(n_dims), 2):
        # 被験者でループ
        for s in range(model.task_size):
            # ★ IDを取得
            s_id = SUBJECT_IDS[s]

            fig, ax = plt.subplots(figsize=(10, 8))
            
            # dim_pair を指定して描画
            plot_utility_contour(ax, model, x_mean, dim_pair, config, s_idx=s)
            
            # ★ タイトルをIDに変更
            title = f"ID {s_id} - Utility Contour\n(Dims {dim_pair[0]} vs {dim_pair[1]})"
            ax.set_title(title, fontsize=config['font_sizes']['title'])
            ax.legend()
            fig.tight_layout()
            
            # ★ ファイル名をIDに変更
            output_filename = f"ID_{s_id}_Dims_{dim_pair[0]}_vs_{dim_pair[1]}.pdf"
            output_path = os.path.join(output_dir_individual, output_filename)
            fig.savefig(output_path) 
            plt.close(fig)
            
    print(f"個人の効用関数（全120ペア * {model.task_size}人）の個別保存が完了しました。")

def plot_pc_reconstructed_contours(model: SparseKPCA, config: dict):
    """PC復元関数（等高線）を個別PDFで保存する"""
    
    output_dir_pc_contours = os.path.join(config['output_dir'], "pc_reconstructed_contours")
    os.makedirs(output_dir_pc_contours, exist_ok=True)
    print(f"\n主成分(PC)から復元した関数（等高線）を '{output_dir_pc_contours}' フォルダに個別保存します...")
    
    if model.Z is None:
        print("警告: model.Z が存在しないため、PCの復元をスキップします。")
        return

    x_mean = config['x_mean']
    n_dims = x_mean.shape[0] # 16次元
    
    z_mean = np.mean(model.Z, axis=0)
    z_std = np.std(model.Z, axis=0)
    
    n_pcs = min(config['model_dim'], model.Z.shape[1])
    
    # 120通りのペアでループ
    for dim_pair in combinations(range(n_dims), 2):
        # PCでループ (PC1からPC20まで)
        for i in range(n_pcs):
            pc_label = f"PC{i+1}"
            sd_val = z_std[i]
            pair_label = f"Dims_{dim_pair[0]}_vs_{dim_pair[1]}"
            
            # --- (1) Mean Function (PCi基準) ---
            z_mean_copy = z_mean.copy()
            z_vec_mean = z_mean_copy[None, :]
            
            fig_mean, ax_mean = plt.subplots(figsize=(10, 8))
            plot_utility_contour(ax_mean, model, x_mean, dim_pair, config, z_vec=z_vec_mean)
            ax_mean.set_title(f"Mean Function (Ref for {pc_label})\n({pair_label})", fontsize=config['font_sizes']['title'])
            ax_mean.legend()
            fig_mean.tight_layout()
            output_path_mean = os.path.join(output_dir_pc_contours, f"{pc_label}_Mean_{pair_label}.pdf")
            fig_mean.savefig(output_path_mean)
            plt.close(fig_mean)

            # --- (2) PCi -3SD ---
            z_minus = z_mean.copy()
            z_minus[i] -= 3 * sd_val
            z_vec_minus = z_minus[None, :]
            
            fig_minus, ax_minus = plt.subplots(figsize=(10, 8))
            plot_utility_contour(ax_minus, model, x_mean, dim_pair, config, z_vec=z_vec_minus)
            ax_minus.set_title(f"{pc_label} -3SD (std={sd_val:.3f})\n({pair_label})", fontsize=config['font_sizes']['title'])
            ax_minus.legend()
            fig_minus.tight_layout()
            output_path_minus = os.path.join(output_dir_pc_contours, f"{pc_label}_Minus3SD_{pair_label}.pdf")
            fig_minus.savefig(output_path_minus)
            plt.close(fig_minus)

            # --- (3) PCi +3SD ---
            z_plus = z_mean.copy()
            z_plus[i] += 3 * sd_val
            z_vec_plus = z_plus[None, :]
            
            fig_plus, ax_plus = plt.subplots(figsize=(10, 8))
            plot_utility_contour(ax_plus, model, x_mean, dim_pair, config, z_vec=z_vec_plus)
            ax_plus.set_title(f"{pc_label} +3SD (std={sd_val:.3f})\n({pair_label})", fontsize=config['font_sizes']['title'])
            ax_plus.legend()
            fig_plus.tight_layout()
            output_path_plus = os.path.join(output_dir_pc_contours, f"{pc_label}_Plus3SD_{pair_label}.pdf")
            fig_plus.savefig(output_path_plus)
            plt.close(fig_plus)

    print(f"主成分(PC1~{n_pcs})から復元した関数（全120ペア * {n_pcs} PC * 3種）の個別保存が完了しました。")


def plot_average_utility_contour(model: SparseKPCA, config: dict):
    """全被験者の平均効用関数（等高線）を個別PDFで保存する。"""
    
    output_dir_average = os.path.join(config['output_dir'], "average_contours")
    os.makedirs(output_dir_average, exist_ok=True)
    print(f"\n全被験者の平均効用関数（等高線）を '{output_dir_average}' フォルダに個別保存します...")
    
    x_mean = config['x_mean']
    n_dims = x_mean.shape[0] # 16次元

    # 120通りのペアでループ
    for dim_pair in combinations(range(n_dims), 2):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # s_idx=None, z_vec=None で平均効用 (W0) を使う
        plot_utility_contour(ax, model, x_mean, dim_pair, config, s_idx=None, z_vec=None)
        
        title = f"Overall Average Utility Contour\n(Dims {dim_pair[0]} vs {dim_pair[1]})"
        ax.set_title(title, fontsize=config['font_sizes']['title'])
        ax.legend()
        fig.tight_layout()
        
        output_filename = f"Average_Utility_Dims_{dim_pair[0]}_vs_{dim_pair[1]}.pdf"
        output_path = os.path.join(output_dir_average, output_filename)
        fig.savefig(output_path) 
        plt.close(fig) 
    
    print(f"全被験者の平均効用関数（全120ペア）の個別保存が完了しました。")


def generate_and_save_pca_plots(model: SparseKPCA, config: dict, params: dict):
    """最終モデルのPCA散布図を個別に保存する (ID名使用)"""
    
    pca_output_path = os.path.join(config['output_dir'], "pca_plots")
    os.makedirs(pca_output_path, exist_ok=True)
    print(f"\n最終モデルのPCA散布図を '{pca_output_path}' フォルダに個別保存します...")
    
    font_sizes = config['font_sizes']
    num_people = config['num_people']
    colors = plt.cm.rainbow(np.linspace(0, 1, num_people))

    if model.Z is None:
        print("警告: model.Z が存在しないため、PCA散布図の作成をスキップします。")
        return

    n_pcs = min(config['model_dim'], model.Z.shape[1])
    
    if n_pcs < 2:
        print("警告: 主成分が2未満のため、PCA散布図の作成をスキップします。")
        return

    base_title_prefix = f"Final Model\nlen={params.get('length', 'N/A'):.3f}, nl={params.get('noise_level', 'N/A'):.4f}"

    # PC1 vs PCi (i=2...n_pcs) のループ
    for i in range(1, n_pcs): 
        pc_y_index = i
        pc_x_index = 0 # PC1固定
        
        pc_x_label = f"PC{pc_x_index + 1}"
        pc_y_label = f"PC{pc_y_index + 1}"
        
        plt.figure(figsize=(10, 10))
        
        pc_x_data = model.Z[:, pc_x_index]
        pc_y_data = model.Z[:, pc_y_index]
        
        plt.scatter(pc_x_data, pc_y_data, c=colors, alpha=0.8)
        
        y_range = np.max(pc_y_data) - np.min(pc_y_data)
        offset = y_range * 0.02  

        for p_idx, z_point in enumerate(model.Z):
            # ★ ラベルをIDに変更
            label_str = str(SUBJECT_IDS[p_idx]) if p_idx < len(SUBJECT_IDS) else str(p_idx+1)
            plt.text(z_point[pc_x_index], z_point[pc_y_index] + offset, label_str, fontsize=9, ha='center')
        
        plot_data = model.Z[:, [pc_x_index, pc_y_index]]
        limit = np.max(np.abs(plot_data)) * 1.1
        plt.xlim(-limit, limit); plt.ylim(-limit, limit)
        
        plt.xlabel(pc_x_label, fontsize=font_sizes['label'])
        plt.ylabel(pc_y_label, fontsize=font_sizes['label'])
        plt.title(base_title_prefix + f"\n({pc_x_label} vs {pc_y_label})", fontsize=font_sizes['title'])
        plt.tick_params(axis='both', which='major', labelsize=font_sizes['ticks'])
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        base_filename = f"final_model_scatter_{pc_x_label}_vs_{pc_y_label}.pdf"
        save_path = os.path.join(pca_output_path, base_filename)
        plt.savefig(save_path)
        plt.close()

    print(f"PCA散布図 (PC1 vs PC2~{n_pcs}) の個別保存が完了しました。")


def generate_and_save_dendrograms(model: SparseKPCA, config: dict, params: dict):
    """最終モデルのデンドログラムを保存する (ID名使用)"""
    print("\n最終モデルのデンドログラムを生成・保存します...")
    dendrogram_output_path = os.path.join(config['output_dir'], "dendrograms")
    os.makedirs(dendrogram_output_path, exist_ok=True)
    
    font_sizes = config['font_sizes']
    num_people = config['num_people']
    num_clusters = 6 

    fig, ax = plt.subplots(figsize=(12, 8))
    if model.a is not None and model.K is not None:
        diff = model.a[:, None, :] - model.a[None, :, :]
        dist_sq = np.einsum('ijn,nm,ijm->ij', diff, model.K, diff)
        dist_mat = np.sqrt(np.maximum(dist_sq, 0.0))
        linked = linkage(squareform(dist_mat, checks=False), method="ward")
        
        if num_clusters > 1 and len(linked) >= num_clusters - 1:
            color_threshold = linked[-(num_clusters - 1), 2] + 1e-6
        else:
            color_threshold = 0

        dendrogram(
            linked,
            ax=ax,
            # ★ ラベルにSUBJECT_IDSを使用
            labels=SUBJECT_IDS,
            color_threshold=color_threshold,
            above_threshold_color='gray' 
        )
        
        ax.set_xlabel("Participant ID", fontsize=font_sizes['label'])
        ax.set_ylabel("Ward distance", fontsize=font_sizes['label'])
        ax.tick_params(axis='both', which='major', labelsize=font_sizes['ticks'])
    
    ax.set_title(f"Final Model\nlen={params.get('length', 'N/A'):.3f}, nl={params.get('noise_level', 'N/A'):.4f}", fontsize=font_sizes['title'])
    fig.tight_layout()
    
    filename = "final_model_dendrogram.pdf"
    save_path = os.path.join(dendrogram_output_path, filename)
    plt.savefig(save_path)
    plt.close(fig)

    print(f"デンドログラムを '{dendrogram_output_path}' フォルダに個別ファイルとして保存しました。")


def display_contribution_ratios(model: SparseKPCA, config: dict):
    """学習済み最終モデルの主成分寄与率と累積寄与率を表示する。"""
    print("\n最終モデルの主成分寄与率:")
    
    if model.eVal is None:
        print("警告: モデルの固有値が計算されていません。")
        return
        
    total_eigenvalue = np.sum(model.eVal)
    contribution_ratios = (model.eVal / total_eigenvalue) * 100
    cumulative_ratios = np.cumsum(contribution_ratios)
    
    print("-" * 60)
    print(f"{'PC':<5} | {'Eigenvalue':<15} | {'Contribution (%)':<18} | {'Cumulative (%)':<18}")
    print("-" * 60)
    
    n_pcs = min(config['model_dim'], len(model.eVal))
    for i in range(n_pcs):
        ratio = contribution_ratios[i]
        cum_ratio = cumulative_ratios[i]
        print(f"PC {i+1:<2} | {model.eVal[i]:<15.4f} | {ratio:<18.2f} | {cum_ratio:<18.2f}")
        
    print("-" * 60)


def train_and_save_final_model(fixed_params: dict, config: dict, x_train: list, y_train: list):
    """固定パラメータで最終モデルを学習し、保存し、可視化する。"""
    print("\n固定パラメータで最終モデルを学習します...")
    
    final_model = SparseKPCA(
        x_list=x_train, 
        y_list=y_train, 
        params=fixed_params, 
        modelDim=config['model_dim'], 
        jitter=config['jitter']
    )
    print("最終モデルの主成分分析 (fit) を実行します...")
    final_model.fit() 
    print("fit() が完了しました。")
    
    model_path = os.path.join(config['output_dir'], "final_model.pkl")
    joblib.dump(final_model, model_path, compress=3)
    print(f"学習済みモデルを '{model_path}' に保存しました。")
    
    # 寄与率
    display_contribution_ratios(final_model, config)
    
    # --- 最終モデルの 2D等高線 可視化関数群を呼び出し ---

    # 1. 個人の効用関数
    plot_individual_utility_contours(final_model, config) 
    
    # 2. PCごとの±3SD
    plot_pc_reconstructed_contours(final_model, config)
    
    # 3. 最終モデルの全被験者平均
    plot_average_utility_contour(final_model, config)
    
    # 4. 最終モデルのPCAとデンドログラム
    if config['model_dim'] >= 2:
        generate_and_save_pca_plots(final_model, config, fixed_params)
        generate_and_save_dendrograms(final_model, config, fixed_params)


def main():
    config = {
        'model_dim': 20,
        'jitter': 1e-6,
        'output_dir': "result_fixed_params", 
        'cv_K': 5, 
        'font_sizes': { 
            'title': 16, 'label': 14, 'ticks': 12, 'legend': 10
        },
        'plot_dim': 1.0,         # 描画範囲 (-1.0 ~ +1.0)
        'grid_points': 40        # グリッドの解像度
    }
    
    # ★ 固定ハイパーパラメータ
    fixed_params = {
        'length': 2.0,  
        'noise_level': 1.0
    }
    
    print(f"固定ハイパーパラメータを使用します: {fixed_params}")

    x_train, y_train, = load_psycho_data2_2.load_data()
    
    config['num_people'] = len(x_train) 
    print(f"\n合計 {config['num_people']} 人のデータを読み込みました。")
    
    all_x = np.concatenate(x_train, axis=0)
    config['x_mean'] = np.mean(all_x, axis=0)
    
    if config['x_mean'].shape[0] != 16:
         print(f"警告: 入力データの次元が16ではありません (次元: {config['x_mean'].shape[0]})")

    os.makedirs(config['output_dir'], exist_ok=True)
    
    train_and_save_final_model(fixed_params, config, x_train, y_train)
    
    print("\n処理が完了しました。")

if __name__ == '__main__':
    main()