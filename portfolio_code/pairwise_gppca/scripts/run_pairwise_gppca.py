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

from pairwise_gppca import DATASET_NAMES, KernelParameters, PairwiseGPPCA, load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise GPPCA on one of the supported preference datasets.")
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="thurstone_pairwise")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--length", type=float, default=0.25)
    parser.add_argument("--preference-noise", type=float, default=1.0)
    parser.add_argument("--basis-size", type=int, default=64)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--newton-max-iter", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num-subjects", type=int, default=None)
    parser.add_argument("--session-count", type=int, default=1)
    parser.add_argument("--min-score-gap", type=float, default=1.0)
    parser.add_argument("--max-comparisons", type=int, default=400)
    parser.add_argument("--grid-size", type=int, default=80)
    return parser.parse_args()


def load_subjects(args: argparse.Namespace):
    if args.dataset == "golden_ratio_induced":
        return load_dataset(
            args.dataset,
            data_dir=args.data_dir,
            num_subjects=args.num_subjects or 21,
            min_score_gap=args.min_score_gap,
            max_comparisons=args.max_comparisons,
            random_state=args.seed,
        )
    if args.dataset == "thurstone_pairwise":
        return load_dataset(
            args.dataset,
            data_dir=args.data_dir,
            session_count=args.session_count,
            subject_count=args.num_subjects or 14,
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def save_basis_outputs(args: argparse.Namespace, model: PairwiseGPPCA) -> None:
    basis_df = pd.DataFrame(model.x_all, columns=[f"x{i + 1}" for i in range(model.input_dim)])
    basis_df.insert(0, "basis_index", np.arange(model.basis_count))
    basis_df.to_csv(args.output_dir / "basis_points.csv", index=False)

    posterior_df = basis_df.copy()
    reconstructed_df = basis_df.copy()
    for subject_index, label in enumerate(model.subject_labels):
        safe_label = label.replace(" ", "_")
        posterior_df[safe_label] = model.posterior_mean[subject_index]
        reconstructed_df[safe_label] = model.reconstructed_mean[subject_index]
    posterior_df.to_csv(args.output_dir / "posterior_mean_basis.csv", index=False)
    reconstructed_df.to_csv(args.output_dir / "reconstructed_mean_basis.csv", index=False)


def save_summary(args: argparse.Namespace, model: PairwiseGPPCA) -> None:
    summary_df = pd.DataFrame(
        {
            "dataset": args.dataset,
            "subject_id": model.subject_ids,
            "subject_label": model.subject_labels,
            "num_stimuli": [subject.stimulus_count for subject in model.subjects],
            "num_pairwise_comparisons": [subject.comparison_count for subject in model.subjects],
            "input_dim": model.input_dim,
            "basis_count": model.basis_count,
        }
    )
    summary_df.to_csv(args.output_dir / "pairwise_summary.csv", index=False)

    latent_df = pd.DataFrame(model.z, columns=[f"z{i + 1}" for i in range(model.z.shape[1])])
    latent_df.insert(0, "subject_label", model.subject_labels)
    latent_df.insert(0, "subject_id", model.subject_ids)
    latent_df.to_csv(args.output_dir / "latent_coordinates.csv", index=False)


def save_one_dimensional_outputs(args: argparse.Namespace, model: PairwiseGPPCA) -> None:
    x_min = min(float(subject.x.min()) for subject in model.subjects)
    x_max = max(float(subject.x.max()) for subject in model.subjects)
    grid = np.linspace(x_min, x_max, args.grid_size)[:, None]
    mean, cov = model.predict_single(grid)
    reconstructed_mean, reconstructed_cov = model.predict_reconstructed(grid)

    posterior_df = pd.DataFrame({"x": grid[:, 0]})
    reconstructed_df = pd.DataFrame({"x": grid[:, 0]})
    for subject_index, label in enumerate(model.subject_labels):
        safe_label = label.replace(" ", "_")
        posterior_df[safe_label] = mean[subject_index]
        reconstructed_df[safe_label] = reconstructed_mean[subject_index]
    posterior_df.to_csv(args.output_dir / "posterior_mean_curves.csv", index=False)
    reconstructed_df.to_csv(args.output_dir / "reconstructed_mean_curves.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        colors = plt.cm.rainbow(np.linspace(0, 1, len(model.subjects)))
        figure, axis = plt.subplots(figsize=(12, 8))
        for subject_index, color in enumerate(colors):
            std = np.sqrt(np.clip(np.diag(cov[subject_index]), 0.0, None))
            axis.plot(grid[:, 0], mean[subject_index], color=color, label=model.subject_labels[subject_index])
            axis.fill_between(
                grid[:, 0],
                mean[subject_index] - 2.0 * std,
                mean[subject_index] + 2.0 * std,
                color=color,
                alpha=0.15,
            )
        axis.set_xlabel("Input")
        axis.set_ylabel("Latent utility")
        axis.set_title(f"Posterior mean curves: {args.dataset}")
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        figure.tight_layout()
        figure.savefig(args.output_dir / "subject_curves.pdf")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(12, 8))
        for subject_index, color in enumerate(colors):
            std = np.sqrt(np.clip(np.diag(reconstructed_cov[subject_index]), 0.0, None))
            axis.plot(grid[:, 0], reconstructed_mean[subject_index], color=color, label=model.subject_labels[subject_index])
            axis.fill_between(
                grid[:, 0],
                reconstructed_mean[subject_index] - 2.0 * std,
                reconstructed_mean[subject_index] + 2.0 * std,
                color=color,
                alpha=0.15,
            )
        axis.set_xlabel("Input")
        axis.set_ylabel("Latent utility")
        axis.set_title(f"GPPCA reconstructed curves: {args.dataset}")
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        figure.tight_layout()
        figure.savefig(args.output_dir / "reconstructed_subject_curves.pdf")
        plt.close(figure)
    except ModuleNotFoundError:
        pass


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_subjects(args)
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

    save_summary(args, model)
    save_basis_outputs(args, model)
    if model.input_dim == 1:
        save_one_dimensional_outputs(args, model)

    with (args.output_dir / "pairwise_gppca_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)


if __name__ == "__main__":
    main()
