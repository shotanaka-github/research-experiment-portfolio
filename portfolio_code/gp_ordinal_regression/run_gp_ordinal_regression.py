from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Gaussian process ordinal regression.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--start-index", type=int, default=10)
    parser.add_argument("--end-index", type=int, default=21)
    parser.add_argument("--lengthscale", type=float, default=0.1)
    return parser.parse_args()


def load_data(data_dir: Path, filename: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(data_dir / f"{filename}.csv", header=0, dtype=float)
    x = frame.iloc[:, 2].to_numpy().reshape(-1, 1)
    y = frame.iloc[:, 1].to_numpy().reshape(-1, 1)
    return x, y


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    x_test = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)

    for index in range(args.start_index, args.end_index + 1):
        filename = f"data{index}"
        x_train, y_train = load_data(args.data_dir, filename)
        model = fit_subject(x_train, y_train, args.lengthscale)
        mean, variance = model.predict_y(x_test)

        subject_df = pd.DataFrame(
            {
                "subject_id": index,
                "x": x_test[:, 0],
                "mean": mean.numpy().reshape(-1),
                "variance": variance.numpy().reshape(-1),
            }
        )
        subject_df.to_csv(args.output_dir / f"{filename}_predictions.csv", index=False)
        all_rows.append(subject_df)

        try:
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(8, 5))
            mu = subject_df["mean"].to_numpy()
            var = subject_df["variance"].to_numpy()
            axis.plot(x_test[:, 0], mu, color="tab:blue")
            axis.fill_between(
                x_test[:, 0],
                mu - 2.0 * np.sqrt(var),
                mu + 2.0 * np.sqrt(var),
                color="tab:blue",
                alpha=0.2,
            )
            axis.scatter(x_train[:, 0], y_train[:, 0], color="tab:red", marker="x")
            axis.set_xlabel("Input")
            axis.set_ylabel("Rating")
            axis.set_title(f"GP ordinal regression: {filename}")
            figure.tight_layout()
            figure.savefig(args.output_dir / f"{filename}_plot.pdf")
            plt.close(figure)
        except ModuleNotFoundError:
            pass

    pd.concat(all_rows, ignore_index=True).to_csv(args.output_dir / "all_predictions.csv", index=False)


if __name__ == "__main__":
    main()
