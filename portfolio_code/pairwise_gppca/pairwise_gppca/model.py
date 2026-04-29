from __future__ import annotations

from dataclasses import dataclass
from math import erf, pi, sqrt

import numpy as np

from .types import SubjectPreferenceData, standardize_vector


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length: float) -> np.ndarray:
    diff = x1[:, None, :] - x2[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    return np.exp(-dist_sq / (2.0 * length**2))


def _normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * values * values) / sqrt(2.0 * pi)


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(values / sqrt(2.0)))


@dataclass(frozen=True)
class KernelParameters:
    length: float
    preference_noise: float = 1.0
    init_noise_variance: float = 0.2
    newton_max_iter: int = 40
    newton_tolerance: float = 1e-6


class PairwiseGPPCA:
    """Pairwise-preference extension of GPPCA with shared basis functions."""

    def __init__(
        self,
        subjects: list[SubjectPreferenceData],
        params: KernelParameters,
        latent_dim: int = 2,
        basis_size: int | None = None,
        jitter: float = 1e-6,
        random_state: int = 0,
    ) -> None:
        if not subjects:
            raise ValueError("subjects must not be empty.")

        self.subjects = subjects
        self.params = params
        self.task_size = len(subjects)
        self.latent_dim = int(latent_dim)
        self.jitter = float(jitter)
        self.random_state = int(random_state)

        self.x_list = [np.asarray(subject.x, dtype=np.float64) for subject in subjects]
        self.initial_target_list = [
            np.zeros(subject.stimulus_count, dtype=np.float64)
            if subject.initial_targets is None
            else standardize_vector(subject.initial_targets)
            for subject in subjects
        ]
        self.subject_ids = np.asarray([subject.subject_id for subject in subjects], dtype=np.int64)
        self.subject_labels = [subject.label or f"subject_{subject.subject_id}" for subject in subjects]
        self.dataset_name = subjects[0].dataset_name or "pairwise_dataset"
        self.input_dim = self.x_list[0].shape[1]

        self.x_all = self._build_basis(self.x_list, basis_size)
        self.basis_count = self.x_all.shape[0]

        self.kernel_matrix = _rbf_kernel(self.x_all, self.x_all, self.params.length)
        self.kernel_matrix += self.jitter * np.eye(self.basis_count)
        self.kernel_inv = np.linalg.inv(self.kernel_matrix)

        self.design_matrices = [
            _rbf_kernel(x_task, self.x_all, self.params.length) @ self.kernel_inv
            for x_task in self.x_list
        ]
        self.posterior_mean, self.posterior_cov = self._compute_task_posteriors()
        self.theta1, self.theta2 = self._to_natural_params(self.posterior_mean, self.posterior_cov)
        self.eta1, self.eta2 = self._to_expectation_params(self.posterior_mean, self.posterior_cov)

        self.z = np.zeros((self.task_size, self.latent_dim), dtype=np.float64)
        self.z_aug = np.concatenate([self.z, np.ones((self.task_size, 1))], axis=1)

        rng = np.random.default_rng(self.random_state)
        self.w1 = rng.uniform(-1e-3, 1e-3, size=(self.latent_dim, self.basis_count))
        upper = np.triu(rng.uniform(0.0, 1e-3, size=(self.latent_dim, self.basis_count, self.basis_count)))
        self.w2 = -0.5 * np.einsum("hij,hkj->hik", upper, upper)

        center_w1, center_w2 = self._legendre_inverse(
            self.eta1.mean(axis=0, keepdims=True),
            self.eta2.mean(axis=0, keepdims=True),
        )
        self.w1 = np.concatenate([self.w1, center_w1], axis=0)
        self.w2 = np.concatenate([self.w2, center_w2], axis=0)

        self.theta_hat1, self.theta_hat2 = self._theta_from_latent(self.z_aug, self.w1, self.w2)
        self.eta_hat1, self.eta_hat2 = self._legendre(self.theta_hat1, self.theta_hat2)
        self.reconstructed_mean, self.reconstructed_cov = self._to_parametric_form(
            self.theta_hat1,
            self.theta_hat2,
        )

    @staticmethod
    def _unique_rows(values: np.ndarray) -> np.ndarray:
        rounded = np.round(values, decimals=12)
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        return values[np.sort(unique_indices)]

    @classmethod
    def _build_basis(cls, x_list: list[np.ndarray], basis_size: int | None) -> np.ndarray:
        x_all = cls._unique_rows(np.concatenate(x_list, axis=0))
        if basis_size is None or basis_size >= x_all.shape[0]:
            return x_all

        if x_all.shape[1] == 1:
            x_min = float(np.min(x_all))
            x_max = float(np.max(x_all))
            return np.linspace(x_min, x_max, int(basis_size))[:, None]

        order = np.lexsort(np.flipud(x_all.T))
        sorted_x = x_all[order]
        chosen = np.linspace(0, sorted_x.shape[0] - 1, int(basis_size), dtype=int)
        return sorted_x[chosen]

    def _initial_posterior_mean(self, design_matrix: np.ndarray, initial_targets: np.ndarray) -> np.ndarray:
        beta = 1.0 / max(self.params.init_noise_variance, 1e-8)
        precision = self.kernel_inv + beta * (design_matrix.T @ design_matrix)
        rhs = beta * (design_matrix.T @ initial_targets)
        return np.linalg.solve(precision, rhs)

    def _log_posterior(self, utility_basis: np.ndarray, comparison_matrix: np.ndarray) -> float:
        scaled = (comparison_matrix @ utility_basis) / self.params.preference_noise
        log_likelihood = np.log(np.clip(_normal_cdf(scaled), 1e-12, 1.0)).sum()
        prior = -0.5 * utility_basis @ (self.kernel_inv @ utility_basis)
        return float(log_likelihood + prior)

    def _laplace_posterior(
        self,
        design_matrix: np.ndarray,
        winner_indices: np.ndarray,
        loser_indices: np.ndarray,
        initial_targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        comparison_matrix = design_matrix[winner_indices] - design_matrix[loser_indices]
        utility_basis = self._initial_posterior_mean(design_matrix, initial_targets)
        current_objective = self._log_posterior(utility_basis, comparison_matrix)

        for _ in range(self.params.newton_max_iter):
            scaled = (comparison_matrix @ utility_basis) / self.params.preference_noise
            cdf = np.clip(_normal_cdf(scaled), 1e-12, 1.0)
            pdf = _normal_pdf(scaled)
            ratio = pdf / cdf
            curvature = ratio * (ratio + scaled) / (self.params.preference_noise**2)

            gradient = -(self.kernel_inv @ utility_basis)
            gradient += comparison_matrix.T @ (ratio / self.params.preference_noise)

            precision = self.kernel_inv + (comparison_matrix.T * curvature) @ comparison_matrix
            precision += self.jitter * np.eye(self.basis_count)

            delta = np.linalg.solve(precision, gradient)
            step_size = 1.0
            updated = utility_basis + step_size * delta
            updated_objective = self._log_posterior(updated, comparison_matrix)

            while updated_objective < current_objective and step_size > 1e-3:
                step_size *= 0.5
                updated = utility_basis + step_size * delta
                updated_objective = self._log_posterior(updated, comparison_matrix)

            if updated_objective < current_objective:
                break

            if np.max(np.abs(updated - utility_basis)) < self.params.newton_tolerance:
                utility_basis = updated
                break

            utility_basis = updated
            current_objective = updated_objective

        scaled = (comparison_matrix @ utility_basis) / self.params.preference_noise
        cdf = np.clip(_normal_cdf(scaled), 1e-12, 1.0)
        pdf = _normal_pdf(scaled)
        ratio = pdf / cdf
        curvature = ratio * (ratio + scaled) / (self.params.preference_noise**2)
        precision = self.kernel_inv + (comparison_matrix.T * curvature) @ comparison_matrix
        precision += self.jitter * np.eye(self.basis_count)
        covariance = np.linalg.inv(precision)
        return utility_basis, covariance

    def _compute_task_posteriors(self) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros((self.task_size, self.basis_count), dtype=np.float64)
        cov = np.zeros((self.task_size, self.basis_count, self.basis_count), dtype=np.float64)

        for task_index, subject in enumerate(self.subjects):
            task_mean, task_cov = self._laplace_posterior(
                design_matrix=self.design_matrices[task_index],
                winner_indices=subject.winner_indices,
                loser_indices=subject.loser_indices,
                initial_targets=self.initial_target_list[task_index],
            )
            mean[task_index] = task_mean
            cov[task_index] = task_cov

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
        sigma = -0.5 * np.linalg.inv(theta2)
        sigma = 0.5 * (sigma + np.swapaxes(sigma, 1, 2))
        mu = np.einsum("tij,tj->ti", sigma, theta1)
        return mu, sigma

    @staticmethod
    def _eta_to_parametric_form(eta1: np.ndarray, eta2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sigma = eta2 - eta1[:, :, None] * eta1[:, None, :]
        sigma = 0.5 * (sigma + np.swapaxes(sigma, 1, 2))
        return eta1, sigma

    def _legendre(self, theta1: np.ndarray, theta2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = self._to_parametric_form(theta1, theta2)
        return self._to_expectation_params(mu, sigma)

    def _legendre_inverse(self, eta1: np.ndarray, eta2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = self._eta_to_parametric_form(eta1, eta2)
        return self._to_natural_params(mu, sigma)

    @staticmethod
    def _theta_from_latent(
        z_aug: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        theta1 = z_aug @ w1
        theta2 = np.einsum("th,hij->tij", z_aug, w2)
        theta2 = 0.5 * (theta2 + np.swapaxes(theta2, 1, 2))
        theta2 -= 1e-6 * np.eye(theta2.shape[-1])[None, :, :]
        return theta1, theta2

    def fit(
        self,
        epochs: int = 120,
        step_z: float = 0.05,
        step_w1: float = 0.2,
        step_w2: float = 0.2,
    ) -> "PairwiseGPPCA":
        for _ in range(int(epochs)):
            delta_eta1 = self.eta_hat1 - self.eta1
            delta_eta2 = self.eta_hat2 - self.eta2

            grad_z = delta_eta1 @ self.w1[: self.latent_dim].T
            grad_z += np.einsum("tij,hij->th", delta_eta2, self.w2[: self.latent_dim])
            self.z -= step_z * grad_z

            self.z_aug = np.concatenate([self.z, np.ones((self.task_size, 1))], axis=1)
            self.w1 -= step_w1 * (self.z_aug.T @ delta_eta1)
            self.w2 -= step_w2 * np.einsum("th,tij->hij", self.z_aug, delta_eta2)
            self.w2 = 0.5 * (self.w2 + np.swapaxes(self.w2, 1, 2))

            self.theta_hat1, self.theta_hat2 = self._theta_from_latent(self.z_aug, self.w1, self.w2)
            self.eta_hat1, self.eta_hat2 = self._legendre(self.theta_hat1, self.theta_hat2)
            self.reconstructed_mean, self.reconstructed_cov = self._to_parametric_form(
                self.theta_hat1,
                self.theta_hat2,
            )

        return self

    def _predict_from_basis_posterior(
        self,
        x_new: np.ndarray,
        basis_mean: np.ndarray,
        basis_cov: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_new = np.asarray(x_new, dtype=np.float64)
        cross_kernel = _rbf_kernel(x_new, self.x_all, self.params.length)
        projection = cross_kernel @ self.kernel_inv
        prior_conditional = _rbf_kernel(x_new, x_new, self.params.length)
        prior_conditional -= projection @ cross_kernel.T
        prior_conditional = 0.5 * (prior_conditional + prior_conditional.T)
        prior_conditional += self.jitter * np.eye(x_new.shape[0])

        mean = np.einsum("nk,tk->tn", projection, basis_mean)
        cov = prior_conditional[None, :, :] + np.einsum("nk,tkl,ml->tnm", projection, basis_cov, projection)
        return mean, cov

    def predict_single(self, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._predict_from_basis_posterior(x_new, self.posterior_mean, self.posterior_cov)

    def predict_reconstructed(self, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._predict_from_basis_posterior(x_new, self.reconstructed_mean, self.reconstructed_cov)

    def predict_preference_probability(
        self,
        x_left: np.ndarray,
        x_right: np.ndarray,
        reconstructed: bool = False,
    ) -> np.ndarray:
        basis_mean = self.reconstructed_mean if reconstructed else self.posterior_mean
        projection_left = _rbf_kernel(np.asarray(x_left, dtype=np.float64), self.x_all, self.params.length) @ self.kernel_inv
        projection_right = _rbf_kernel(np.asarray(x_right, dtype=np.float64), self.x_all, self.params.length) @ self.kernel_inv
        difference = projection_left - projection_right
        utility_gap = np.einsum("nk,tk->tn", difference, basis_mean)
        return _normal_cdf(utility_gap / self.params.preference_noise)
