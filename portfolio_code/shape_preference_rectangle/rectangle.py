import numpy as np
import pandas as pd
import os
import joblib
from itertools import product
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns

import load_psycho_data
from model.KPCA5 import SparseKPCA

# --- 1. モデル評価・ハイパーパラメータ探索 ---

def grid_search_cv(x_list: list, y_list: list, param_grid: dict, cv: int, model_dim: int, jitter: float) -> tuple:
    """共通基底を用いたK-Fold CVでグリッドサーチを実行し、最適なパラメータを探す。"""
    print("グリッドサーチを開始します...")
    keys = list(param_grid.keys())
    param_combinations = [dict(zip(keys, v)) for v in product(*param_grid.values())]
    results = []
    num_subjects = len(x_list)
    x_source_data = x_list[0]

    for i, params in enumerate(param_combinations):
        subject_scores = [[] for _ in range(num_subjects)]
        kf_splits = [list(KFold(n_splits=cv, shuffle=True, random_state=i).split(x)) for i, x in enumerate(x_list)]

        for fold_idx in range(cv):
            train_indices_union = set()
            for subject_idx in range(num_subjects):
                train_idx, _ = kf_splits[subject_idx][fold_idx]
                train_indices_union.update(train_idx)
            
            x_common_basis = x_source_data[sorted(list(train_indices_union))]
            
            x_train_fold, y_train_fold = [], []
            x_val_sets, y_val_sets = {}, {}
            for subject_idx in range(num_subjects):
                train_idx, val_idx = kf_splits[subject_idx][fold_idx]
                x_train_fold.append(x_source_data[train_idx])
                y_train_fold.append(y_list[subject_idx][train_idx])
                x_val_sets[subject_idx] = x_source_data[val_idx]
                y_val_sets[subject_idx] = y_list[subject_idx][val_idx]

            model = SparseKPCA(params, modelDim=model_dim, jitter=jitter)
            model.fit(x_train_fold, y_train_fold, basis_x=x_common_basis)

            for subject_idx in range(num_subjects):
                x_val, y_val = x_val_sets[subject_idx], y_val_sets[subject_idx]
                if x_val.shape[0] > 0:
                    y_pred = model.predict(x_val)[subject_idx, :]
                    mse = mean_squared_error(y_val.flatten(), y_pred.flatten())
                    subject_scores[subject_idx].append(mse)

        total_mse = np.sum([np.mean(scores) for scores in subject_scores if scores])
        results.append({'params': params, 'mean_mse': total_mse})
        print(f"\rグリッドサーチ進行中... {i+1}/{len(param_combinations)}", end="")

    print("\nグリッドサーチが完了しました。")
    if not results:
        return None, None

    best_result = min(results, key=lambda x: x['mean_mse'])
    return best_result['params'], results

# --- 2. 結果の可視化 ---

def generate_and_save_pca_plots(results_df: pd.DataFrame, x_train: list, y_train: list, model_dim: int, jitter: float, output_dir: str):
    """性能ランク別のPCA散布図を生成し、PDFに保存する。"""
    pdf_path = os.path.join(output_dir, "pca_plots_by_rank.pdf")
    selected_results = results_df.iloc[np.linspace(0, len(results_df) - 1, 10, dtype=int)]

    with PdfPages(pdf_path) as pdf:
        for df_index, row in selected_results.iterrows():
            params = {'length': row['length'], 'noise_level': row['noise_level']}
            model = SparseKPCA(params, modelDim=model_dim, jitter=jitter)
            model.fit(x_train, y_train)
            
            plt.figure(figsize=(8, 8))
            if model.Z is not None and model.Z.shape[1] >= 2:
                plt.scatter(model.Z[:, 0], model.Z[:, 1], alpha=0.8)
                for i, (x, y) in enumerate(model.Z):
                    plt.text(x, y, str(i + 1), fontsize=9)
                
                limit = np.max(np.abs(model.Z)) * 1.1
                plt.xlim(-limit, limit); plt.ylim(-limit, limit)
                plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
                plt.grid(True, linestyle='--', alpha=0.6)
            
            title = f"Rank {df_index + 1}\nlength={params['length']:.3f}, noise_level={params['noise_level']:.4f}\nMSE = {row['mean_mse']:.5f}"
            plt.title(title, fontsize=12)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"PCA散布図を '{pdf_path}' に保存しました。")

def generate_and_save_dendrograms(results_df: pd.DataFrame, x_train: list, y_train: list, model_dim: int, jitter: float, output_dir: str, num_people: int):
    """性能ランク別のデンドログラムを生成し、PDFに保存する。"""
    pdf_path = os.path.join(output_dir, "dendrograms_by_rank.pdf")
    selected_results = results_df.iloc[np.linspace(0, len(results_df) - 1, 10, dtype=int)]

    with PdfPages(pdf_path) as pdf:
        for df_index, row in selected_results.iterrows():
            params = {'length': row['length'], 'noise_level': row['noise_level']}
            model = SparseKPCA(params, modelDim=model_dim, jitter=jitter)
            model.fit(x_train, y_train)

            fig, ax = plt.subplots(figsize=(12, 7))
            if model.a is not None and model.K is not None:
                diff = model.a[:, None, :] - model.a[None, :, :]
                dist_sq = np.einsum('ijn,nm,ijm->ij', diff, model.K, diff)
                dist_mat = np.sqrt(np.maximum(dist_sq, 0.0))
                linked = linkage(squareform(dist_mat, checks=False), method="ward")
                dendrogram(linked, ax=ax, labels=np.arange(1, num_people + 1))
                ax.set_xlabel("Subject ID"); ax.set_ylabel("Ward distance")
            
            title = f"Rank {df_index + 1}\nlength={params['length']:.3f}, noise_level={params['noise_level']:.4f}, MSE = {row['mean_mse']:.5f}"
            ax.set_title(title, fontsize=12)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"デンドログラムを '{pdf_path}' に保存しました。")


def plot_and_save_all_predicted_functions(model, output_dir: str, num_people: int):
    """学習済み最終モデルを使い、全被験者の予測関数を重ねてプロットしPDFに保存する。"""
    x_range_for_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
    predicted_functions = model.predict(x_range_for_plot)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, num_people))
    for i in range(num_people):
        plt.plot(x_range_for_plot, predicted_functions[i, :], color=colors[i], alpha=0.7)
        
    plt.title(f'Predicted Functions for All {num_people} Subjects'); plt.xlabel('Input value (x)'); plt.ylabel('Predicted output value (y)')
    plt.xlim(-1, 1); plt.grid(True, linestyle='--', alpha=0.6)
    
    pdf_path = os.path.join(output_dir, "all_predicted_functions.pdf")
    plt.savefig(pdf_path); plt.close()
    print(f"全被験者の予測関数グラフを '{pdf_path}' に保存しました。")

# --- 3. メイン処理の実行管理 ---

def process_and_visualize_results(cv_results: list, config: dict, x_train: list, y_train: list):
    """グリッドサーチの結果を処理し、CSV保存と可視化を行う。"""
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 結果をDataFrameに変換してCSVに保存
    results_df = pd.concat([
        pd.json_normalize([res['params'] for res in cv_results]),
        pd.DataFrame([res['mean_mse'] for res in cv_results], columns=['mean_mse'])
    ], axis=1).sort_values(by='mean_mse').reset_index(drop=True)
    
    csv_path = os.path.join(config['output_dir'], "cv_results.csv")
    results_df.to_csv(csv_path, index_label='rank')
    print(f"\n交差検証の全結果を '{csv_path}' に保存しました。")
    print("--- 上位5件の結果 ---\n", results_df.head())

    # ヒートマップの作成
    pivot_table = results_df.pivot(index='noise_level', columns='length', values='mean_mse')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=False, cmap="viridis_r")
    plt.title("Hyperparameter Grid Search (Total MSE)")
    plt.xlabel("Kernel Length"); plt.ylabel("Noise Level (alpha)")
    heatmap_path = os.path.join(config['output_dir'], "cv_results_heatmap.png")
    plt.savefig(heatmap_path); plt.close()
    print(f"結果のヒートマップを '{heatmap_path}' に保存しました。")

    
    generate_and_save_pca_plots(results_df, x_train, y_train, config['model_dim'], config['jitter'], config['output_dir'])
    generate_and_save_dendrograms(results_df, x_train, y_train, config['model_dim'], config['jitter'], config['output_dir'], config['num_people'])

def train_and_save_final_model(best_params: dict, config: dict, x_train: list, y_train: list):
    """最適なパラメータで最終モデルを学習し、保存する。"""
    print("\n最適なパラメータで最終モデルを再学習します...")
    final_model = SparseKPCA(best_params, modelDim=config['model_dim'], jitter=config['jitter'])
    final_model.fit(x_train, y_train)

    model_path = os.path.join(config['output_dir'], "final_model.pkl")
    joblib.dump(final_model, model_path, compress=3)
    print(f"学習済みモデルを '{model_path}' に保存しました。")
    
    plot_and_save_all_predicted_functions(final_model, config['output_dir'], config['num_people'])

def main():
    """スクリプト全体の実行を管理する。"""
    # --- 設定 ---
    config = {
        'model_dim': 2,
        'jitter': 1e-4,
        'output_dir': "result_renctangle",
        'num_people': 20,
    }
    param_grid = {
        'length': np.logspace(-1, 0, 100),
        'noise_level': np.logspace(-2, 0, 10)
    }

    # --- データ準備 ---
    x_train, y_train = load_psycho_data.load_data()
    # モデルが列ベクトルを期待するため形状を(N, 1)に統一
    #y_train = [y.reshape(-1, 1) for y in y_train_raw]

    # --- グリッドサーチ実行 ---
    best_params, cv_results = grid_search_cv(
        x_train, y_train, param_grid,
        cv=5, model_dim=config['model_dim'], jitter=config['jitter']
    )
    
    # --- 結果処理と最終モデル学習 ---
    if cv_results:
        process_and_visualize_results(cv_results, config, x_train, y_train)
        if best_params:
            train_and_save_final_model(best_params, config, x_train, y_train)
    else:
        print("\nグリッドサーチで有効な結果が得られませんでした。")

if __name__ == '__main__':
    main()