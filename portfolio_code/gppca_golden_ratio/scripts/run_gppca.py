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

from gppca_golden_ratio import GPEPCA, KernelParameters, load_golden_ratio_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPPCA on the golden-ratio dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--length", type=float, default=0.25)
    parser.add_argument("--noise-variance", type=float, default=0.01)
    parser.add_argument("--basis-size", type=int, default=64)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def save_plots(
    output_dir: Path,
    x_list: list[np.ndarray],
    mean: np.ndarray,
    cov: np.ndarray,
) -> None:
    grid = np.linspace(
        min(float(x_subject.min()) for x_subject in x_list),
        max(float(x_subject.max()) for x_subject in x_list),
        mean.shape[1],
    )[:, None]

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(12, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(x_list)))
        for subject_index, color in enumerate(colors):
            std = np.sqrt(np.clip(np.diag(cov[subject_index]), 0.0, None))
            axis.plot(grid[:, 0], mean[subject_index], color=color, label=f"subject {subject_index + 1}")
            axis.fill_between(
                grid[:, 0],
                mean[subject_index] - 2.0 * std,
                mean[subject_index] + 2.0 * std,
                color=color,
                alpha=0.15,
            )

        axis.set_xlabel("Rectangle aspect ratio")
        axis.set_ylabel("Standardized preference score")
        axis.set_title("Posterior mean and uncertainty for each subject")
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "subject_curves.pdf")
        plt.close(figure)
    except ModuleNotFoundError:
        pass


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_list, y_list = load_golden_ratio_data(args.data_dir)
    params = KernelParameters(length=args.length, noise_variance=args.noise_variance)
    model = GPEPCA(
        x_list=x_list,
        y_list=y_list,
        params=params,
        latent_dim=args.latent_dim,
        basis_size=args.basis_size,
        jitter=args.jitter,
        random_state=args.seed,
    ).fit(epochs=args.epochs)

    grid = np.linspace(
        min(float(x_subject.min()) for x_subject in x_list),
        max(float(x_subject.max()) for x_subject in x_list),
        200,
    )[:, None]
    mean, cov = model.predict_single(grid)
    save_plots(args.output_dir, x_list, mean, cov)

    latent_df = pd.DataFrame(model.z, columns=[f"z{i + 1}" for i in range(model.z.shape[1])])
    latent_df.insert(0, "subject_id", np.arange(1, len(latent_df) + 1))
    latent_df.to_csv(args.output_dir / "latent_coordinates.csv", index=False)
    pd.DataFrame(
        [{"length": params.length, "noise_variance": params.noise_variance}]
    ).to_csv(args.output_dir / "selected_hyperparameters.csv", index=False)

    with (args.output_dir / "gppca_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)


if __name__ == "__main__":
    main()
