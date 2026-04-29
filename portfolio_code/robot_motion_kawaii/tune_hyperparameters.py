from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import time

import numpy as np
import pandas as pd

import load_psycho_data2_2
from model.KPCA5 import SparseKPCA


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    std = float(values.std())
    if std <= 1e-12:
        return values - values.mean()
    return (values - values.mean()) / std


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean(diff * diff)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)) ** 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validated hyperparameter search for the robot motion KRR model."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("hyperparameter_search"))
    parser.add_argument("--model-dim", type=int, default=20)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--length-min", type=float, default=10 ** (-0.3))
    parser.add_argument("--length-max", type=float, default=10 ** 0.9)
    parser.add_argument("--length-points", type=int, default=7)
    parser.add_argument("--noise-min", type=float, default=10 ** (-2.0))
    parser.add_argument("--noise-max", type=float, default=10 ** 0.5)
    parser.add_argument("--noise-points", type=int, default=7)
    return parser.parse_args()


def make_kfold_splits(
    n_samples: int,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_samples < n_splits:
        raise ValueError(f"Cannot split {n_samples} samples into {n_splits} folds.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    splits = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        val_indices = np.sort(indices[start:stop])
        train_indices = np.sort(np.concatenate([indices[:start], indices[stop:]]))
        splits.append((train_indices, val_indices))
        start = stop
    return splits


def cross_validated_score(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    params: dict[str, float],
    model_dim: int,
    jitter: float,
    folds: int,
    random_state: int,
    search_index: int | None = None,
    search_total: int | None = None,
) -> float:
    tag = (
        f"[{search_index:03d}/{search_total:03d}] "
        if search_index is not None and search_total is not None
        else ""
    )
    splitters = [
        make_kfold_splits(x_subject.shape[0], folds, random_state + subject_index)
        for subject_index, x_subject in enumerate(x_list)
    ]
    fold_shape_scores: list[float] = []
    fold_mse_scores: list[float] = []

    for fold_index in range(folds):
        fold_started_at = time.perf_counter()
        x_train_fold: list[np.ndarray] = []
        y_train_fold: list[np.ndarray] = []
        x_val_fold: list[np.ndarray] = []
        y_val_fold: list[np.ndarray] = []

        for subject_index, (x_subject, y_subject) in enumerate(zip(x_list, y_list)):
            train_indices, val_indices = splitters[subject_index][fold_index]
            x_train_fold.append(x_subject[train_indices])
            y_train_fold.append(y_subject[train_indices])
            x_val_fold.append(x_subject[val_indices])
            y_val_fold.append(y_subject[val_indices])

        model = SparseKPCA(
            x_list=x_train_fold,
            y_list=y_train_fold,
            params=params,
            modelDim=model_dim,
            jitter=jitter,
        )

        subject_shape_scores = []
        subject_mse_scores = []
        for subject_index, x_val in enumerate(x_val_fold):
            predictions = model.predict_subject(x_val, subject_index)
            subject_mse_scores.append(mean_squared_error(y_val_fold[subject_index], predictions))
            subject_shape_scores.append(
                root_mean_squared_error(
                    zscore(y_val_fold[subject_index]),
                    zscore(predictions),
                )
            )
        fold_shape_mean = float(np.mean(subject_shape_scores))
        fold_mse_mean = float(np.mean(subject_mse_scores))
        fold_shape_scores.append(fold_shape_mean)
        fold_mse_scores.append(fold_mse_mean)
        fold_elapsed = time.perf_counter() - fold_started_at
        print(
            f"{tag}fold {fold_index + 1}/{folds}: "
            f"shape_rmse={fold_shape_mean:.6f}, mean_mse={fold_mse_mean:.6f}, "
            f"elapsed={fold_elapsed/60:.1f} min",
            flush=True,
        )

    return {
        "mean_rmse": float(np.mean(fold_shape_scores)),
        "mean_mse": float(np.mean(fold_mse_scores)),
    }


def search_hyperparameters(
    output_dir: str | Path = "hyperparameter_search",
    model_dim: int = 20,
    jitter: float = 1e-6,
    folds: int = 5,
    random_state: int = 0,
    length_grid: np.ndarray | None = None,
    noise_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_list, y_list = load_psycho_data2_2.load_data()
    if length_grid is None:
        length_grid = np.logspace(-0.3, 0.9, 7)
    if noise_grid is None:
        noise_grid = np.logspace(-2.0, 0.5, 7)
    results = []
    param_grid = list(product(length_grid, noise_grid))
    started_at = time.perf_counter()
    print(
        f"{len(param_grid)} 通りのパラメータ組を探索します "
        f"(length={len(length_grid)}, noise={len(noise_grid)}, folds={folds})。",
        flush=True,
    )

    for index, (length, noise_level) in enumerate(param_grid, start=1):
        params = {"length": float(length), "noise_level": float(noise_level)}
        score_summary = cross_validated_score(
            x_list=x_list,
            y_list=y_list,
            params=params,
            model_dim=model_dim,
            jitter=jitter,
            folds=folds,
            random_state=random_state,
            search_index=index,
            search_total=len(param_grid),
        )
        results.append(
            {
                "length": length,
                "noise_level": noise_level,
                "mean_rmse": score_summary["mean_rmse"],
                "mean_mse": score_summary["mean_mse"],
            }
        )
        elapsed = time.perf_counter() - started_at
        mean_seconds = elapsed / index
        remaining = mean_seconds * (len(param_grid) - index)
        best_so_far = min(results, key=lambda row: row["mean_rmse"])
        print(
            (
                f"[{index:03d}/{len(param_grid):03d}] "
                f"length={length:.6g}, noise={noise_level:.6g}, "
                f"rmse={score_summary['mean_rmse']:.6f}, "
                f"mean_mse={score_summary['mean_mse']:.6f}, "
                f"best=({best_so_far['length']:.6g}, {best_so_far['noise_level']:.6g}) "
                f"rmse_score={best_so_far['mean_rmse']:.6f}, "
                f"elapsed={elapsed/60:.1f} min, eta={remaining/60:.1f} min"
            ),
            flush=True,
        )
        results_df = pd.DataFrame(results).sort_values("mean_rmse").reset_index(drop=True)
        results_df.to_csv(output_dir / "cv_results.csv", index=False)
        results_df.head(10).to_csv(output_dir / "top10_results.csv", index=False)
        results_df.head(1).to_csv(output_dir / "selected_hyperparameters.csv", index=False)

    results_df = pd.DataFrame(results).sort_values("mean_rmse").reset_index(drop=True)
    return results_df


if __name__ == "__main__":
    args = parse_args()
    search_hyperparameters(
        output_dir=args.output_dir,
        model_dim=args.model_dim,
        jitter=args.jitter,
        folds=args.folds,
        random_state=args.random_state,
        length_grid=np.logspace(np.log10(args.length_min), np.log10(args.length_max), args.length_points),
        noise_grid=np.logspace(np.log10(args.noise_min), np.log10(args.noise_max), args.noise_points),
    )
