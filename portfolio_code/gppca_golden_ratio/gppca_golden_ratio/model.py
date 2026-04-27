from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KernelParameters:
    length: float
    noise_variance: float


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length: float) -> np.ndarray:
    diff = x1[:, None, :] - x2[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    return np.exp(-dist_sq / (2.0 * length**2))


class GPEPCA:
    """Gaussian-process exponential-family PCA for subject-specific functions.

    The model first estimates a Gaussian posterior over function weights for each
    subject on a shared basis, then learns a low-dimensional latent description
    of those posteriors in natural-parameter space.
    """

    def __init__(
        self,
        x_list: list[np.ndarray],
        y_list: list[np.ndarray],
        params: KernelParameters,
        latent_dim: int = 2,
        basis_size: int | None = None,
        jitter: float = 1e-6,
        random_state: int = 0,
    ) -> None:
        if len(x_list) != len(y_list):
            raise ValueError("x_list and y_list must have the same length.")

        self.x_list = [np.asarray(x, dtype=np.float64) for x in x_list]
        self.y_list = [np.asarray(y, dtype=np.float64).reshape(-1) for y in y_list]
        self.task_size = len(self.x_list)
        self.params = params
        self.latent_dim = int(latent_dim)
        self.jitter = float(jitter)
        self.random_state = int(random_state)

        self.x_all = self._build_basis(self.x_list, basis_size)
        self.basis_count = self.x_all.shape[0]

        self.kernel_matrix = _rbf_kernel(self.x_all, self.x_all, self.params.length)
        self.kernel_matrix += self.jitter * np.eye(self.basis_count)
        self.kernel_inv = np.linalg.inv(self.kernel_matrix)

        self.posterior_mean, self.posterior_cov = self._compute_task_posteriors()
        self.theta1, self.theta2 = self._to_natural_params(self.posterior_mean, self.posterior_cov)
        self.eta1, self.eta2 = self._to_expectation_params(self.posterior_mean, self.posterior_cov)

        self.z = np.zeros((self.task_size, self.latent_dim), dtype=np.float64)
        self.z_aug = np.concatenate([self.z, np.ones((self.task_size, 1))], axis=1)

        rng = np.random.default_rng(self.random_state)
        self.w1 = rng.uniform(-1e-3, 1e-3, size=(self.latent_dim, self.basis_count))
        u = np.triu(rng.uniform(0.0, 1e-3, size=(self.latent_dim, self.basis_count, self.basis_count)))
        self.w2 = -0.5 * np.einsum("hij,hkj->hik", u, u)

        center_w1, center_w2 = self._legendre_inverse(
            self.eta1.mean(axis=0, keepdims=True),
            self.eta2.mean(axis=0, keepdims=True),
        )
        self.w1 = np.concatenate([self.w1, center_w1], axis=0)
        self.w2 = np.concatenate([self.w2, center_w2], axis=0)

        self.theta_hat1, self.theta_hat2 = self._theta_from_latent(self.z_aug, self.w1, self.w2)
        self.eta_hat1, self.eta_hat2 = self._legendre(self.theta_hat1, self.theta_hat2)
        self.reconstructed_mean, self.reconstructed_cov = self._to_parametric_form(
            self.theta_hat1, self.theta_hat2
        )

    @staticmethod
    def _build_basis(x_list: list[np.ndarray], basis_size: int | None) -> np.ndarray:
        x_all = np.concatenate(x_list, axis=0)
        if basis_size is None or basis_size >= x_all.shape[0]:
            return x_all

        x_min = np.min(x_all, axis=0)
        x_max = np.max(x_all, axis=0)
        if x_all.shape[1] != 1:
            raise ValueError("This cleaned GPPCA script expects one-dimensional inputs.")
        grid = np.linspace(x_min.item(), x_max.item(), int(basis_size))[:, None]
        return grid

    def _compute_task_posteriors(self) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros((self.task_size, self.basis_count), dtype=np.float64)
        cov = np.zeros((self.task_size, self.basis_count, self.basis_count), dtype=np.float64)
        beta = 1.0 / self.params.noise_variance

        for task_index, (x_task, y_task) in enumerate(zip(self.x_list, self.y_list)):
            k_xb = _rbf_kernel(x_task, self.x_all, self.params.length)
            precision = self.kernel_matrix + beta * (k_xb.T @ k_xb)
            precision_inv = np.linalg.inv(precision)
            rhs = beta * (k_xb.T @ y_task)
            mean[task_index] = precision_inv @ rhs
            cov[task_index] = precision_inv

        return mean, cov

    @staticmethod
    def _to_natural_params(mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sigma_inv = np.linalg.inv(sigma)
        theta1 = np.einsum("tij,tj->ti", sigma_inv, mu)
        theta2 = -0.5 * sigma_inv
        return theta1, theta2

    @staticmethod
    def _to_expectation_params(mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eta1 = mu
        eta2 = mu[:, :, None] * mu[:, None, :] + sigma
        return eta1, eta2

    @staticmethod
    def _to_parametric_form(theta1: np.ndarray, theta2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta2_inv = np.linalg.inv(theta2)
        sigma = -0.5 * theta2_inv
        mu = np.einsum("tij,tj->ti", sigma, theta1)
        return mu, sigma

    @staticmethod
    def _eta_to_parametric_form(eta1: np.ndarray, eta2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sigma = eta2 - eta1[:, :, None] * eta1[:, None, :]
        return eta1, sigma

    def _legendre(self, theta1: np.ndarray, theta2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = self._to_parametric_form(theta1, theta2)
        return self._to_expectation_params(mu, sigma)

    def _legendre_inverse(self, eta1: np.ndarray, eta2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = self._eta_to_parametric_form(eta1, eta2)
        return self._to_natural_params(mu, sigma)

    @staticmethod
    def _theta_from_latent(
        z_aug: np.ndarray, w1: np.ndarray, w2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        theta1 = z_aug @ w1
        theta2 = np.einsum("th,hij->tij", z_aug, w2)
        return theta1, theta2

    def fit(
        self,
        epochs: int = 200,
        step_z: float = 0.1,
        step_w1: float = 1.0,
        step_w2: float = 1.0,
    ) -> "GPEPCA":
        for _ in range(int(epochs)):
            delta_eta1 = self.eta_hat1 - self.eta1
            delta_eta2 = self.eta_hat2 - self.eta2

            grad_z = delta_eta1 @ self.w1[: self.latent_dim].T
            grad_z += np.einsum("tij,hij->th", delta_eta2, self.w2[: self.latent_dim])
            self.z -= step_z * grad_z

            self.z_aug = np.concatenate([self.z, np.ones((self.task_size, 1))], axis=1)
            self.w1 -= step_w1 * (self.z_aug.T @ delta_eta1)
            self.w2 -= step_w2 * np.einsum("th,tij->hij", self.z_aug, delta_eta2)

            self.theta_hat1, self.theta_hat2 = self._theta_from_latent(self.z_aug, self.w1, self.w2)
            self.eta_hat1, self.eta_hat2 = self._legendre(self.theta_hat1, self.theta_hat2)
            self.reconstructed_mean, self.reconstructed_cov = self._to_parametric_form(
                self.theta_hat1, self.theta_hat2
            )

        return self

    def predict_single(self, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_new = np.asarray(x_new, dtype=np.float64)
        k_xx = _rbf_kernel(x_new, x_new, self.params.length)
        k_xb = _rbf_kernel(x_new, self.x_all, self.params.length)
        mean = np.einsum("nk,tk->tn", k_xb, self.posterior_mean)
        cov = (k_xx - k_xb @ self.kernel_inv @ k_xb.T)[None, :, :]
        cov = cov + np.einsum("nk,tkl,ml->tnm", k_xb, self.posterior_cov, k_xb)
        return mean, cov

    def predict_reconstructed(self, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_new = np.asarray(x_new, dtype=np.float64)
        k_xx = _rbf_kernel(x_new, x_new, self.params.length)
        k_xb = _rbf_kernel(x_new, self.x_all, self.params.length)
        mean = np.einsum("nk,tk->tn", k_xb, self.reconstructed_mean)
        cov = (k_xx - k_xb @ self.kernel_inv @ k_xb.T)[None, :, :]
        cov = cov + np.einsum("nk,tkl,ml->tnm", k_xb, self.reconstructed_cov, k_xb)
        return mean, cov

