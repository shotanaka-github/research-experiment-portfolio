from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

try:
    from joblib import Parallel, delayed
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Parallel = None
    delayed = None


@dataclass(frozen=True)
class KernelRidgePCAParameters:
    length: float
    noise_level: float


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length: float) -> np.ndarray:
    x1_norm = np.sum(x1 * x1, axis=1)[:, None]
    x2_norm = np.sum(x2 * x2, axis=1)[None, :]
    dist_sq = np.maximum(x1_norm + x2_norm - 2.0 * (x1 @ x2.T), 0.0)
    return np.exp(-dist_sq / (2.0 * length**2))


class CommonBasisKernelRidgePCA:
    """Kernel-ridge regression PCA with a shared basis across subjects.

    The key point is that the subjects do not need to share the same training
    inputs. Each subject is projected onto a common basis `x_all`, and the
    basis weights are then summarized with kernel PCA.
    """

    def __init__(
        self,
        x_list: list[np.ndarray],
        y_list: list[np.ndarray],
        params: KernelRidgePCAParameters,
        latent_dim: int = 2,
        basis_size: int | None = None,
        jitter: float = 1e-6,
        random_state: int = 0,
    ) -> None:
        if len(x_list) != len(y_list):
            raise ValueError("x_list and y_list must have the same length.")

        self.x_list = [np.asarray(x, dtype=np.float64) for x in x_list]
        self.y_list = [np.asarray(y, dtype=np.float64).reshape(-1) for y in y_list]
        self.params = params
        self.latent_dim = int(latent_dim)
        self.jitter = float(jitter)
        self.random_state = int(random_state)

        self.x_all = self._build_basis(self.x_list, basis_size, self.random_state)
        self.basis_count = self.x_all.shape[0]

        self.kernel_matrix = _rbf_kernel(self.x_all, self.x_all, self.params.length)
        self.kernel_matrix += self.jitter * np.eye(self.basis_count)
        self.kernel_cholesky = np.linalg.cholesky(self.kernel_matrix)

        self.weights = self._estimate_all_subject_weights()
        self.weight_mean = None
        self.latent_scores = None
        self.components = None
        self.reconstructed_weights = None
        self.eigenvalues = None
        self.explained_variance_ratio = None

    @staticmethod
    def _build_basis(
        x_list: list[np.ndarray], basis_size: int | None, random_state: int
    ) -> np.ndarray:
        x_all = np.concatenate(x_list, axis=0)
        if basis_size is None or basis_size >= x_all.shape[0]:
            return x_all

        rng = np.random.default_rng(random_state)
        indices = rng.choice(x_all.shape[0], size=int(basis_size), replace=False)
        return x_all[indices]

    def _solve_kernel_system(self, rhs: np.ndarray) -> np.ndarray:
        temp = np.linalg.solve(self.kernel_cholesky, rhs)
        return np.linalg.solve(self.kernel_cholesky.T, temp)

    def _estimate_subject_weights(self, x_subject: np.ndarray, y_subject: np.ndarray) -> np.ndarray:
        beta = 1.0 / self.params.noise_level
        k_xb = _rbf_kernel(x_subject, self.x_all, self.params.length)
        z = self._solve_kernel_system(k_xb.T)
        s = k_xb @ z
        m = s + self.params.noise_level * np.eye(s.shape[0])
        v1 = z @ y_subject
        v2 = k_xb @ v1
        correction = np.linalg.solve(m, v2)
        return beta * (v1 - z @ correction)

    def _estimate_all_subject_weights(self) -> np.ndarray:
        weights = np.zeros((len(self.x_list), self.basis_count), dtype=np.float64)
        for index, (x_subject, y_subject) in enumerate(zip(self.x_list, self.y_list)):
            weights[index] = self._estimate_subject_weights(x_subject, y_subject)
        return weights

    def fit(self) -> "CommonBasisKernelRidgePCA":
        self.weight_mean = self.weights.mean(axis=0)
        centered = self.weights - self.weight_mean
        gram = centered @ self.kernel_matrix @ centered.T

        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        valid = eigenvalues > 1e-10
        eigenvalues = eigenvalues[valid]
        eigenvectors = eigenvectors[:, valid]
        effective_dim = min(self.latent_dim, len(eigenvalues))

        self.eigenvalues = eigenvalues
        total = eigenvalues.sum()
        self.explained_variance_ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)

        if effective_dim == 0:
            self.latent_scores = np.zeros((len(self.x_list), 0))
            self.components = np.zeros((0, self.basis_count))
            self.reconstructed_weights = np.repeat(self.weight_mean[None, :], len(self.x_list), axis=0)
            return self

        self.latent_scores = eigenvectors[:, :effective_dim] @ np.diag(np.sqrt(eigenvalues[:effective_dim]))
        self.components = (
            np.diag(1.0 / eigenvalues[:effective_dim]) @ self.latent_scores.T @ centered
        )
        self.reconstructed_weights = self.latent_scores @ self.components + self.weight_mean
        return self

    def predict_subject(self, x_new: np.ndarray, subject_index: int) -> np.ndarray:
        k_xb = _rbf_kernel(np.asarray(x_new, dtype=np.float64), self.x_all, self.params.length)
        return k_xb @ self.weights[int(subject_index)]

    def predict_reconstructed_subject(self, x_new: np.ndarray, subject_index: int) -> np.ndarray:
        if self.reconstructed_weights is None:
            raise RuntimeError("Call fit() before using reconstructed predictions.")
        k_xb = _rbf_kernel(np.asarray(x_new, dtype=np.float64), self.x_all, self.params.length)
        return k_xb @ self.reconstructed_weights[int(subject_index)]

    def generate_from_latent(self, x_new: np.ndarray, latent_scores: np.ndarray) -> np.ndarray:
        if self.components is None or self.weight_mean is None:
            raise RuntimeError("Call fit() before generating functions from the latent space.")
        latent_scores = np.asarray(latent_scores, dtype=np.float64)
        if latent_scores.ndim == 1:
            latent_scores = latent_scores[None, :]
        weights = latent_scores @ self.components + self.weight_mean
        k_xb = _rbf_kernel(np.asarray(x_new, dtype=np.float64), self.x_all, self.params.length)
        return np.einsum("nk,ik->in", k_xb, weights)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff * diff)))


def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(diff * diff))


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    std = float(values.std())
    if std <= 1e-12:
        return values - values.mean()
    return (values - values.mean()) / std


def _iter_kfold_indices(
    n_samples: int, n_splits: int, random_state: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_samples < n_splits:
        raise ValueError(f"n_samples={n_samples} is smaller than n_splits={n_splits}.")

    indices = np.arange(n_samples)
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    splits = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_index = indices[start:stop]
        train_index = np.concatenate([indices[:start], indices[stop:]])
        splits.append((train_index, val_index))
        current = stop
    return splits


def _build_subject_folds(
    x_list: list[np.ndarray], y_list: list[np.ndarray], n_splits: int, random_state: int
) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    subject_folds: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = []
    for x_subject, y_subject in zip(x_list, y_list):
        subject_records = []
        for train_index, val_index in _iter_kfold_indices(
            n_samples=len(x_subject),
            n_splits=n_splits,
            random_state=random_state,
        ):
            subject_records.append(
                (
                    x_subject[train_index],
                    y_subject[train_index],
                    x_subject[val_index],
                    y_subject[val_index],
                )
            )
        subject_folds.append(subject_records)
    return subject_folds


def _score_parameter_set(
    params: KernelRidgePCAParameters,
    subject_folds: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    latent_dim: int,
    basis_size: int | None,
    jitter: float,
    random_state: int,
) -> dict[str, float]:
    fold_shape_scores = []
    fold_mse_scores = []
    n_splits = len(subject_folds[0])

    for fold_index in range(n_splits):
        x_train = [subject_folds[s][fold_index][0] for s in range(len(subject_folds))]
        y_train = [subject_folds[s][fold_index][1] for s in range(len(subject_folds))]
        x_val = [subject_folds[s][fold_index][2] for s in range(len(subject_folds))]
        y_val = [subject_folds[s][fold_index][3] for s in range(len(subject_folds))]

        model = CommonBasisKernelRidgePCA(
            x_list=x_train,
            y_list=y_train,
            params=params,
            latent_dim=latent_dim,
            basis_size=basis_size,
            jitter=jitter,
            random_state=random_state,
        )

        subject_shape_scores = []
        subject_mse_scores = []
        for subject_index, (x_subject_val, y_subject_val) in enumerate(zip(x_val, y_val)):
            prediction = model.predict_subject(x_subject_val, subject_index)
            subject_mse_scores.append(_mean_squared_error(y_subject_val, prediction))
            subject_shape_scores.append(_rmse(_zscore(y_subject_val), _zscore(prediction)))
        fold_shape_scores.append(float(np.mean(subject_shape_scores)))
        fold_mse_scores.append(float(np.mean(subject_mse_scores)))

    return {
        "length": float(params.length),
        "noise_level": float(params.noise_level),
        "mean_rmse": float(np.mean(fold_shape_scores)),
        "mean_mse": float(np.mean(fold_mse_scores)),
    }


def search_hyperparameters(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    lengths: np.ndarray,
    noise_levels: np.ndarray,
    n_splits: int = 5,
    latent_dim: int = 2,
    basis_size: int | None = None,
    jitter: float = 1e-6,
    random_state: int = 42,
    n_jobs: int = 1,
) -> tuple[KernelRidgePCAParameters, list[dict[str, float]]]:
    subject_folds = _build_subject_folds(x_list, y_list, n_splits=n_splits, random_state=random_state)
    param_grid = [
        KernelRidgePCAParameters(length=float(length), noise_level=float(noise_level))
        for length, noise_level in product(lengths, noise_levels)
    ]

    if n_jobs == 1:
        results = [
            _score_parameter_set(
                params=params,
                subject_folds=subject_folds,
                latent_dim=latent_dim,
                basis_size=basis_size,
                jitter=jitter,
                random_state=random_state,
            )
            for params in param_grid
        ]
    else:
        if Parallel is None or delayed is None:
            raise ModuleNotFoundError("joblib is required when n_jobs is not 1.")
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_score_parameter_set)(
                params=params,
                subject_folds=subject_folds,
                latent_dim=latent_dim,
                basis_size=basis_size,
                jitter=jitter,
                random_state=random_state,
            )
            for params in param_grid
        )

    best = min(results, key=lambda row: row["mean_rmse"])
    best_params = KernelRidgePCAParameters(length=best["length"], noise_level=best["noise_level"])
    return best_params, results
