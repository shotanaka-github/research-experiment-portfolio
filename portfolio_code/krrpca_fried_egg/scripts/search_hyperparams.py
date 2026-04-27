from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from krrpca_fried_egg import load_fried_egg_data, search_hyperparameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter search for KRRPCA on fried-egg data.")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--label-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/hyperparams"))
    parser.add_argument("--length-min", type=float, default=0.1)
    parser.add_argument("--length-max", type=float, default=1.0)
    parser.add_argument("--length-points", type=int, default=5)
    parser.add_argument("--noise-min", type=float, default=0.01)
    parser.add_argument("--noise-max", type=float, default=1.0)
    parser.add_argument("--noise-points", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--basis-size", type=int, default=256)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_list, y_list, _ = load_fried_egg_data(args.feature_dir, args.label_dir)
    lengths = np.logspace(np.log10(args.length_min), np.log10(args.length_max), args.length_points)
    noise_levels = np.logspace(np.log10(args.noise_min), np.log10(args.noise_max), args.noise_points)

    best_params, results = search_hyperparameters(
        x_list=x_list,
        y_list=y_list,
        lengths=lengths,
        noise_levels=noise_levels,
        n_splits=args.cv_folds,
        latent_dim=args.latent_dim,
        basis_size=args.basis_size,
        jitter=args.jitter,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    results_df = pd.DataFrame(results).sort_values("mean_rmse").reset_index(drop=True)
    results_df.to_csv(args.output_dir / "cv_results.csv", index=False)

    with (args.output_dir / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"length": best_params.length, "noise_level": best_params.noise_level},
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
