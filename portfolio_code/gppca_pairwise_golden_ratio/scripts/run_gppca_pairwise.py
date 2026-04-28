from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppca_pairwise_golden_ratio import (  # noqa: E402
    KernelParameters,
    PairwiseGPPCA,
    load_pairwise_golden_ratio_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise GPPCA on the golden-ratio dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--num-subjects", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--length", type=float, default=0.25)
    parser.add_argument("--preference-noise", type=float, default=1.0)
    parser.add_argument("--min-score-gap", type=float, default=1.0)
    parser.add_argument("--max-comparisons", type=int, default=400)
    parser.add_argument("--basis-size", type=int, default=32)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--newton-max-iter", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_pairwise_golden_ratio_data(
        data_dir=args.data_dir,
        num_subjects=args.num_subjects,
        min_score_gap=args.min_score_gap,
        max_comparisons=args.max_comparisons,
        random_state=args.seed,
    )
    model = PairwiseGPPCA(
        subjects=subjects,
        params=KernelParameters(
            length=args.length,
            preference_noise=args.preference_noise,
            newton_max_iter=args.newton_max_iter,
        ),
        latent_dim=args.latent_dim,
        basis_size=args.basis_size,
        jitter=args.jitter,
        random_state=args.seed,
    ).fit(epochs=args.epochs)

    x_min = min(float(subject.x.min()) for subject in subjects)
    x_max = max(float(subject.x.max()) for subject in subjects)
    grid = np.linspace(x_min, x_max, 200)[:, None]
    mean, cov = model.predict_single(grid)
    reconstructed_mean, reconstructed_cov = model.predict_reconstructed(grid)

    try:
        import matplotlib.pyplot as plt

        colors = plt.cm.rainbow(np.linspace(0, 1, len(subjects)))

        figure, axis = plt.subplots(figsize=(12, 8))
        for subject_index, color in enumerate(colors):
            std = np.sqrt(np.clip(np.diag(cov[subject_index]), 0.0, None))
            axis.plot(grid[:, 0], mean[subject_index], color=color, label=f"subject {subjects[subject_index].subject_id}")
            axis.fill_between(
                grid[:, 0],
                mean[subject_index] - 2.0 * std,
                mean[subject_index] + 2.0 * std,
                color=color,
                alpha=0.15,
            )

        axis.set_xlabel("Rectangle aspect ratio")
        axis.set_ylabel("Latent utility")
        axis.set_title("Pairwise GP posterior mean and uncertainty for each subject")
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        figure.tight_layout()
        figure.savefig(args.output_dir / "subject_curves.pdf")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(12, 8))
        for subject_index, color in enumerate(colors):
            std = np.sqrt(np.clip(np.diag(reconstructed_cov[subject_index]), 0.0, None))
            axis.plot(
                grid[:, 0],
                reconstructed_mean[subject_index],
                color=color,
                label=f"subject {subjects[subject_index].subject_id}",
            )
            axis.fill_between(
                grid[:, 0],
                reconstructed_mean[subject_index] - 2.0 * std,
                reconstructed_mean[subject_index] + 2.0 * std,
                color=color,
                alpha=0.15,
            )

        axis.set_xlabel("Rectangle aspect ratio")
        axis.set_ylabel("Latent utility")
        axis.set_title("GPPCA-reconstructed latent utility curves")
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        figure.tight_layout()
        figure.savefig(args.output_dir / "reconstructed_subject_curves.pdf")
        plt.close(figure)
    except ModuleNotFoundError:
        pass

    latent_df = pd.DataFrame(model.z, columns=[f"z{i + 1}" for i in range(model.z.shape[1])])
    latent_df.insert(0, "subject_id", model.subject_ids)
    latent_df.to_csv(args.output_dir / "latent_coordinates.csv", index=False)

    comparison_df = pd.DataFrame(
        {
            "subject_id": model.subject_ids,
            "num_observations": [subject.x.shape[0] for subject in subjects],
            "num_pairwise_comparisons": [subject.comparison_count for subject in subjects],
            "mean_score": [float(subject.scores.mean()) for subject in subjects],
        }
    )
    comparison_df.to_csv(args.output_dir / "pairwise_summary.csv", index=False)

    posterior_df = pd.DataFrame({"x": grid[:, 0]})
    reconstructed_df = pd.DataFrame({"x": grid[:, 0]})
    for subject_index, subject in enumerate(subjects):
        posterior_df[f"subject_{subject.subject_id}"] = mean[subject_index]
        reconstructed_df[f"subject_{subject.subject_id}"] = reconstructed_mean[subject_index]
    posterior_df.to_csv(args.output_dir / "posterior_mean_curves.csv", index=False)
    reconstructed_df.to_csv(args.output_dir / "reconstructed_mean_curves.csv", index=False)

    with (args.output_dir / "gppca_pairwise_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)


if __name__ == "__main__":
    main()
