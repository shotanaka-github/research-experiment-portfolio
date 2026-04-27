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

from krrpca_fried_egg import CommonBasisKernelRidgePCA, KernelRidgePCAParameters, load_fried_egg_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit KRRPCA on fried-egg data.")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--label-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/final"))
    parser.add_argument("--length", type=float, default=0.3)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--basis-size", type=int, default=256)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_list, y_list, subject_ids = load_fried_egg_data(args.feature_dir, args.label_dir)
    model = CommonBasisKernelRidgePCA(
        x_list=x_list,
        y_list=y_list,
        params=KernelRidgePCAParameters(length=args.length, noise_level=args.noise_level),
        latent_dim=args.latent_dim,
        basis_size=args.basis_size,
        jitter=args.jitter,
        random_state=args.random_state,
    ).fit()

    latent_df = pd.DataFrame(
        model.latent_scores,
        columns=[f"z{i + 1}" for i in range(model.latent_scores.shape[1])],
    )
    latent_df.insert(0, "subject_id", subject_ids)
    latent_df.to_csv(args.output_dir / "latent_coordinates.csv", index=False)

    variance_df = pd.DataFrame(
        {
            "component": np.arange(1, len(model.explained_variance_ratio) + 1),
            "eigenvalue": model.eigenvalues,
            "explained_variance_ratio": model.explained_variance_ratio,
        }
    )
    variance_df.to_csv(args.output_dir / "explained_variance.csv", index=False)

    if model.latent_scores.shape[1] >= 2:
        try:
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(8, 8))
            colors = np.where(subject_ids <= 20, "tab:blue", "tab:red")
            axis.scatter(model.latent_scores[:, 0], model.latent_scores[:, 1], c=colors)
            for subject_id, x_coord, y_coord in zip(
                subject_ids, model.latent_scores[:, 0], model.latent_scores[:, 1]
            ):
                axis.text(x_coord, y_coord, str(subject_id), fontsize=8, ha="left", va="bottom")
            axis.set_xlabel("Latent dimension 1")
            axis.set_ylabel("Latent dimension 2")
            axis.set_title("KRRPCA latent space")
            figure.tight_layout()
            figure.savefig(args.output_dir / "latent_space.pdf")
            plt.close(figure)
        except ModuleNotFoundError:
            pass

    with (args.output_dir / "krrpca_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)


if __name__ == "__main__":
    main()
