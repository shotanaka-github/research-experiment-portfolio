from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from load_ordinal_data import load_ordinal_data
from model.KROR_PCA2 import SparseKPCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit kernel ordinal regression PCA.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--length", type=float, default=0.15)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--basis-size", type=int, default=10)
    parser.add_argument("--jitter", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_ordinal_data(args.data_dir)
    model = SparseKPCA(
        x_train,
        y_train,
        {"length": args.length, "noise_level": args.noise_level},
        modelDim=args.latent_dim,
        basis_size=args.basis_size,
        jitter=args.jitter,
    )
    model.fit()

    latent_df = pd.DataFrame(
        model.Z,
        columns=[f"z{i + 1}" for i in range(model.Z.shape[1])],
    )
    latent_df.insert(0, "subject_id", np.arange(1, model.Z.shape[0] + 1))
    latent_df.to_csv(args.output_dir / "latent_coordinates.csv", index=False)

    threshold_df = pd.DataFrame(
        model.alpha_est,
        columns=[f"threshold_{i + 1}" for i in range(model.alpha_est.shape[1])],
    )
    threshold_df.insert(0, "subject_id", np.arange(1, model.alpha_est.shape[0] + 1))
    threshold_df.to_csv(args.output_dir / "estimated_thresholds.csv", index=False)

    grid = np.linspace(-1.0, 1.0, 200)[:, None]
    predicted = model.predict(grid)
    predicted_df = pd.DataFrame(predicted.T)
    predicted_df.insert(0, "x", grid[:, 0])
    predicted_df.to_csv(args.output_dir / "predicted_latent_functions.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        colors = plt.cm.rainbow(np.linspace(0, 1, predicted.shape[0]))
        figure, axis = plt.subplots(figsize=(10, 8))
        for subject_index, color in enumerate(colors):
            axis.plot(grid[:, 0], predicted[subject_index], color=color, alpha=0.7)
        axis.set_xlabel("Input")
        axis.set_ylabel("Latent utility")
        axis.set_title("Kernel ordinal regression latent functions")
        figure.tight_layout()
        figure.savefig(args.output_dir / "predicted_functions_all_subjects.pdf")
        plt.close(figure)
    except ModuleNotFoundError:
        pass

    with (args.output_dir / "kernel_ordinal_regression.pkl").open("wb") as handle:
        pickle.dump(model, handle)


if __name__ == "__main__":
    main()
