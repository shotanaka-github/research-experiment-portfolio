from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppca_golden_ratio import GPEPCA, KernelParameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ordinal GP + GPPCA on the golden-ratio dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=21)
    parser.add_argument("--lengthscale", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--basis-size", type=int, default=64)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def load_data(data_dir: Path, filename: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(data_dir / f"{filename}.csv", header=0, dtype=float)
    x = frame.iloc[:, 2].to_numpy().reshape(-1, 1)
    y = frame.iloc[:, 1].to_numpy().reshape(-1, 1)
    return x, y


def build_basis(x_list: list[np.ndarray], basis_size: int) -> np.ndarray:
    x_all = np.concatenate(x_list, axis=0)
    x_min = float(x_all.min())
    x_max = float(x_all.max())
    return np.linspace(x_min, x_max, int(basis_size), dtype=np.float64).reshape(-1, 1)


def fit_subject(x: np.ndarray, y: np.ndarray, lengthscale: float):
    import gpflow

    bin_edges = np.arange(7, dtype=float) + 1.0
    bin_edges = bin_edges - bin_edges.mean()
    likelihood = gpflow.likelihoods.Ordinal(bin_edges)
    kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscale)
    model = gpflow.models.VGP(data=(x, y), kernel=kernel, likelihood=likelihood)

    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss,
        variables=model.trainable_variables,
        options={"disp": False, "maxiter": 100},
    )
    return model


def estimate_posteriors(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    x_basis: np.ndarray,
    lengthscale: float,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    posterior_mean = np.zeros((len(x_list), x_basis.shape[0]), dtype=np.float64)
    posterior_cov = np.zeros((len(x_list), x_basis.shape[0], x_basis.shape[0]), dtype=np.float64)

    for subject_index, (x_train, y_train) in enumerate(zip(x_list, y_list)):
        model = fit_subject(x_train, y_train, lengthscale)
        mean, variance = model.predict_f(x_basis)
        posterior_mean[subject_index] = mean.numpy().reshape(-1)
        posterior_cov[subject_index] = np.diag(np.clip(variance.numpy().reshape(-1), jitter, None))

    return posterior_mean, posterior_cov


def save_plots(
    output_dir: Path,
    x_basis: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_cov: np.ndarray,
    reconstructed_mean: np.ndarray,
    reconstructed_cov: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt

        colors = plt.cm.rainbow(np.linspace(0, 1, posterior_mean.shape[0]))

        figure, axis = plt.subplots(figsize=(12, 8))
        for subject_index, color in enumerate(colors):
            std = np.sqrt(np.clip(np.diag(posterior_cov[subject_index]), 0.0, None))
            axis.plot(x_basis[:, 0], posterior_mean[subject_index], color=color, label=f"subject {subject_index + 1}")
            axis.fill_between(
                x_basis[:, 0],
                posterior_mean[subject_index] - 2.0 * std,
                posterior_mean[subject_index] + 2.0 * std,
                color=color,
                alpha=0.15,
            )
        axis.set_xlabel("Rectangle aspect ratio")
        axis.set_ylabel("Latent ordinal GP mean")
        axis.set_title("Ordinal GP posterior mean on shared basis")
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "ordinal_gp_posterior_curves.pdf")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(12, 8))
        for subject_index, color in enumerate(colors):
            std = np.sqrt(np.clip(np.diag(reconstructed_cov[subject_index]), 0.0, None))
            axis.plot(x_basis[:, 0], reconstructed_mean[subject_index], color=color, label=f"subject {subject_index + 1}")
            axis.fill_between(
                x_basis[:, 0],
                reconstructed_mean[subject_index] - 2.0 * std,
                reconstructed_mean[subject_index] + 2.0 * std,
                color=color,
                alpha=0.15,
            )
        axis.set_xlabel("Rectangle aspect ratio")
        axis.set_ylabel("Latent ordinal GP mean")
        axis.set_title("Ordinal GPPCA reconstructed curves")
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "ordinal_gppca_reconstructed_curves.pdf")
        plt.close(figure)
    except ModuleNotFoundError:
        pass


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    subject_ids: list[int] = []
    for index in range(args.start_index, args.end_index + 1):
        filename = f"data{index}"
        x_train, y_train = load_data(args.data_dir, filename)
        x_list.append(x_train)
        y_list.append(y_train)
        subject_ids.append(index)

    x_basis = build_basis(x_list, args.basis_size)
    posterior_mean, posterior_cov = estimate_posteriors(
        x_list=x_list,
        y_list=y_list,
        x_basis=x_basis,
        lengthscale=args.lengthscale,
        jitter=args.jitter,
    )

    params = KernelParameters(length=args.lengthscale, noise_variance=args.jitter)
    model = GPEPCA.from_posteriors(
        x_all=x_basis,
        posterior_mean=posterior_mean,
        posterior_cov=posterior_cov,
        params=params,
        latent_dim=args.latent_dim,
        jitter=args.jitter,
        random_state=args.seed,
    ).fit(epochs=args.epochs)

    latent_df = pd.DataFrame(model.z, columns=[f"z{i + 1}" for i in range(model.z.shape[1])])
    latent_df.insert(0, "subject_id", subject_ids)
    latent_df.to_csv(args.output_dir / "latent_coordinates.csv", index=False)

    basis_df = pd.DataFrame({"x": x_basis[:, 0]})
    posterior_df = basis_df.copy()
    reconstructed_df = basis_df.copy()
    for subject_index, subject_id in enumerate(subject_ids):
        label = f"subject_{subject_id}"
        posterior_df[label] = posterior_mean[subject_index]
        reconstructed_df[label] = model.reconstructed_mean[subject_index]
    posterior_df.to_csv(args.output_dir / "posterior_mean_basis.csv", index=False)
    reconstructed_df.to_csv(args.output_dir / "reconstructed_mean_basis.csv", index=False)

    save_plots(
        output_dir=args.output_dir,
        x_basis=x_basis,
        posterior_mean=posterior_mean,
        posterior_cov=posterior_cov,
        reconstructed_mean=model.reconstructed_mean,
        reconstructed_cov=model.reconstructed_cov,
    )

    with (args.output_dir / "ordinal_gppca_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)


if __name__ == "__main__":
    main()
