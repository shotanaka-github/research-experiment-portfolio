import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
import jax
from jax import vmap, grad
from jax.scipy.special import erf  # erf を使ってProbitを定義
import random

class SparseKPCA(object):
    def __init__(self, x_list, y_list, params, modelDim = 1, basis_size=10, jitter=0):
        # inputs
        self.task_size = len(x_list)
        self.x_list = x_list
        self.y_list = y_list
        self.basis_size = basis_size
        if basis_size is None:
            self.x_all = np.concatenate(self.x_list, axis=0)
        else:
            x_min = np.concatenate(x_list, axis=0).min()
            x_max = np.concatenate(x_list, axis=0).max()
            node, step = np.linspace(x_min, x_max, self.basis_size, retstep=True)
            self.x_all = node[:, None]

        self.N, self.D = self.x_all.shape
        self.kernel = self.rbf
        self.params = params
        self.jitter = jitter
        self.L = modelDim
        
        # カーネル行列
        self.K = self.kernel(self.x_all, self.x_all, self.params["length"]) + self.jitter * np.eye(self.N)
        self.Kinv = np.linalg.inv(self.K)

        # 順序回帰用の重みを学習 (probit)
        self.a = self.calc_weight_ordinal()  # 順序probitの枠組みによる学習結果(各タスク)
        

    def calc_weight_ordinal(self, max_iter=500, lr=1e-5):#lr（学習率）の値が高すぎると発散するため注意が必要
        num_class = 7       # クラス数
        num_th = num_class - 1  # 閾値の数(=6)
        
        # 学習結果を格納
        a_est = np.zeros((self.task_size, self.N))  
        alpha_est = np.zeros((self.task_size, num_th))  # 各タスクでの閾値

        # ridge の正則化係数
        lambda_reg = 1.0 / self.params["noise_level"]
        
        # 標準正規CDF (Probit) を定義
        def probit(x):
            return 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))
        
        # 負の対数尤度 + リッジ正則化を定義
        def loss_fn(params, Ki, y):
            a_i = params[:self.N]
            alpha_raw = params[self.N:]
            # alphaをsortして単調増加を担保
            alpha = jnp.sort(alpha_raw)  # (6,)

            # 潜在関数値 f = Ki @ a_i
            f = Ki @ a_i  # (Ni,)

            def log_prob_ordinal(yi, fi):
                # alpha_0 = -∞, alpha_7 = +∞ とみなす
                alpha_ext = jnp.concatenate([
                    jnp.array([-jnp.inf]),
                    alpha,
                    jnp.array([jnp.inf])
                ])  # shape=(8,)

                # yiカテゴリの場合:
                #   P(Y=yi) = Φ(alpha[yi] - fi) - Φ(alpha[yi-1] - fi) /つけなくて
                lower = probit(alpha_ext[yi - 1] - fi)
                upper = probit(alpha_ext[yi] - fi)

                p = jnp.clip(upper - lower, 1e-10, 1.0)  # underflow防止
                return jnp.log(p)

            # 尤度 (データ平均)
            nll_data = - jnp.mean(vmap(log_prob_ordinal)(y, f))

            # 正則化項 (a_i^T K a_i) に lambda_reg をかける例
            reg = 0.5 * lambda_reg * (a_i @ (self.K @ a_i))

            return nll_data + reg

        # タスクごとに最適化
        for i in range(self.task_size):
            x_i = self.x_list[i]
            y_i = self.y_list[i]  # {1,2,3,4,5,6,7}想定
            Ki = self.kernel(x_i, self.x_all, self.params["length"])
            
            # 初期値
            init_a = jnp.zeros((self.N,))
            init_alpha = jnp.linspace(-2.0, 2.0, num_th)  # 例: -2〜2を均等割り
            init_params = jnp.concatenate([init_a, init_alpha])
            params = init_params  # ★ここが重要


            def update(params, lr=lr):
                g = grad(loss_fn)(params, Ki, y_i)
                return params - lr*g

            print(f"--- Task {i+1}/{self.task_size} ---")
            # 進捗表示: tqdmを使い、毎回の loss を出してみる
            for it in tqdm(range(max_iter), desc=f"Gradient descent task={i+1}"):
                params = update(params, lr=lr)
                # もし途中経過のlossを表示したい場合
                if (it % 50 == 0) or (it == max_iter-1):
                    current_loss = loss_fn(params, Ki, y_i)
                    # ここはJAXの値なので numpy化
                    current_loss_val = float(current_loss)
                    print(f"  Iter {it:4d}, loss={current_loss_val:.4f}")
            # 学習済みを格納
            a_i_opt = params[:self.N]
            alpha_i_opt = jnp.sort(params[self.N:])
            a_est[i] = np.array(a_i_opt)
            alpha_est[i] = np.array(alpha_i_opt)
            print("a_推定",a_est)
            print("aのshapeshapeshape",a_est.shape)

        self.alpha_est = alpha_est  # 各タスクの閾値
        return a_est

    def fit(self):
        # a (タスクx基底点)を用いてPCA
        self.W0 = self.a.mean(axis=0)
        S = (self.a - self.W0[None, :]) @ self.K @ (self.a - self.W0[None, :]).T
        eVal, eVec = np.linalg.eigh(S)
        index = np.argsort(-eVal)
        
        self.eVal = eVal[index]
        self.eVec = eVec[:, index]
        
        self.Z = self.eVec[:, :self.L] @ np.diag(np.sqrt(self.eVal[:self.L]))
        self.W = np.diag(1.0 / self.eVal[:self.L]) @ self.Z.T @ (self.a - self.W0[None, :])
        self.a_pos = self.Z @ self.W + self.W0

    def predict(self, x_new):
        Kn = self.kernel(x_new, self.x_all, self.params["length"])
        mu = np.einsum("nk,ik->in", Kn, self.a)
        print()
        return mu

    def predict_approx(self, x_new):
        Kn = self.kernel(x_new, self.x_all, self.params["length"])
        mu = np.einsum("nk, ik->in", Kn, self.a_pos)
        return mu

    def generate_pos(self, x_new, Z):
        a = Z @ self.W + self.W0
        Kn = self.kernel(x_new, self.x_all, self.params["length"])
        mu = np.einsum("nk, ik->in", Kn, a)
        return mu

    def rbf(self, x1, x2, length):
        dist = ((x1[:, None, :] - x2[None, :, :])**2).sum(axis=2)
        return 2 * np.exp(- dist / (2 * length**2))
    
    def predict_y(self, x_new):
        f_all = self.predict(x_new)  # shape=(N_new, task_size)
        N_new = f_all.shape[0]
        y_pred = np.zeros((N_new, self.task_size), dtype=int)

        for t in range(self.task_size):
            alpha_task = self.alpha_est[t]  # shape=(6,)
            alpha_ext = np.concatenate(([-np.inf], alpha_task, [np.inf]))
            f_task = f_all[:, t]  # shape=(N_new,)
            cat_t = np.searchsorted(alpha_ext, f_task, side="right") - 1
            y_pred[:, t] = cat_t

        return y_pred
