# model/KPCA.py

import numpy as np

class SparseKPCA(object):
    """
    scikit-learnのAPIスタイルに合わせたSparseKPCAクラス。
    1. __init__でハイパーパラメータを保持
    2. fit(X, y)でモデルを学習
    3. predict(X_new)で予測
    """
    def __init__(self, params, modelDim=1, jitter=1e-4):
        """モデルのハイパーパラメータを初期化"""
        # ハイパーパラメータ
        self.params = params
        self.L = modelDim
        self.jitter = jitter
        self.kernel = self.rbf

        # fitメソッドで学習される属性
        self.task_size = None
        self.x_list = None
        self.y_list = None
        self.x_all = None
        self.N = None
        self.D = None
        self.K = None
        self.Kinv = None
        self.a = None
        self.W0 = None
        self.eVal = None
        self.eVec = None
        self.Z = None
        self.W = None
        self.a_pos = None

# model/KPCA5.py

    def fit(self, x_list, y_list, basis_x=None):
        """
        与えられたデータでモデルを学習させる。
        （カーネルリッジ回帰による重み計算と、カーネルPCAによる次元削減）
        
        Args:
            x_list (list): 各タスクの入力データ(x)のリスト
            y_list (list): 各タスクの出力データ(y)のリスト
            basis_x (np.ndarray, optional): 共通基底として使用するデータ。
                                            Noneの場合はx_listの全データから生成。
        """
        # STEP 1: データのセットアップと共通基底の構築
        self.task_size = len(x_list)
        self.x_list = x_list
        self.y_list = y_list

        if basis_x is not None:
            self.x_all = basis_x
        else:
            # --- ▼▼▼ 修正箇所 ▼▼▼ ---
            # basis_xが指定されない場合は、入力された全データ点の「和集合(unique)」を基底とする
            # 1次元にフラット化 -> unique (ソート済み) -> reshape(-1, 1) で (N, 1) の形状に戻す
            all_x_flat = np.concatenate([x.ravel() for x in self.x_list])
            
            # Xが1次元であることを前提とし、(N, 1) の形状に戻す
            self.x_all = np.unique(all_x_flat).reshape(-1, 1)
            # --- ▲▲▲ 修正箇所 ▲▲▲ ---
        
        self.N, self.D = self.x_all.shape
        self.K = self.kernel(self.x_all, self.x_all, self.params["length"]) + self.jitter * np.eye(self.N)
        self.Kinv = np.linalg.inv(self.K)

        # STEP 2: カーネルリッジ回帰で各タスクの重み 'a' を計算
        self.a = self._calc_weight()

        # STEP 3: カーネルPCAで重み 'a' を低次元空間に射影
        self.W0 = self.a.mean(axis=0)
        S = (self.a - self.W0[None, :]) @ self.K @ (self.a - self.W0[None, :]).T
        eVal, eVec = np.linalg.eigh(S)
        index = np.argsort(-eVal)
        
        self.eVal = eVal[index]
        self.eVec = eVec[:, index]
        
        # 低次元表現と再構成のための重みを計算
        self.Z = self.eVec[:, :self.L] @ np.diag(np.sqrt(self.eVal[:self.L]))
        self.W = np.diag(1.0 / self.eVal[:self.L]) @ self.Z.T @ (self.a - self.W0[None, :])
        self.a_pos = self.Z @ self.W + self.W0

        return self

    def _calc_weight(self):
        """【内部関数】カーネルリッジ回帰の重みを計算"""
        a = np.zeros((self.task_size, self.N))
        beta = 1.0 / self.params["noise_level"]
        for i in range(self.task_size):
            Ki = self.kernel(self.x_list[i], self.x_all, self.params["length"])
            Pi = self.K + (beta * Ki.T @ Ki)
            Piinv = np.linalg.inv(Pi)
            
            # --- ▼▼▼ 前回のエラー修正箇所 (形状を (N,) に合わせる) ▼▼▼ ---
            p = beta * Ki.T @ self.y_list[i].ravel() # .ravel() が必要
            # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
            
            a[i] = Piinv @ p      
        return a

    def predict(self, x_new):
        """新しい入力データに対して予測を行う"""
        Kn = self.kernel(x_new, self.x_all, self.params["length"])
        mu = np.einsum("nk, ik->in", Kn, self.a)
        return mu

    def predict_approx(self, x_new):
        """低次元表現から再構成された重みを用いて予測を行う"""
        Kn = self.kernel(x_new, self.x_all, self.params["length"])
        mu = np.einsum("nk, ik->in", Kn, self.a_pos)
        return mu

    def generate_pos(self, x_new, Z):
        """指定された潜在変数Zから対応する関数値を生成する"""
        a = Z @ self.W + self.W0
        Kn = self.kernel(x_new, self.x_all, self.params["length"])
        mu = np.einsum("nk, ik->in", Kn, a)
        return mu

    def rbf(self, x1, x2, length):
        """RBFカーネル関数"""
        dist = ((x1[:, None, :] - x2[None, :, :])**2).sum(axis=2)
        # 係数2は元コードに合わせていますが、通常は不要なことが多いです
        return 2 * np.exp(- dist / (2 * length**2))