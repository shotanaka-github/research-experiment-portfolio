from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import load_psycho_data2_2


def zscore_tensor(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True, unbiased=False)
    safe_std = torch.where(std > 1e-12, std, torch.ones_like(std))
    centered = values - mean
    return torch.where(std > 1e-12, centered / safe_std, centered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated cross-validated hyperparameter search for the robot motion KRR model."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("hyperparameter_search_gpu"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--length-min", type=float, default=1.0)
    parser.add_argument("--length-max", type=float, default=5.0)
    parser.add_argument("--length-points", type=int, default=7)
    parser.add_argument("--noise-min", type=float, default=1e-2)
    parser.add_argument("--noise-max", type=float, default=1.0)
    parser.add_argument("--noise-points", type=int, default=7)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--basis-size", type=int, default=None)
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


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float64":
        return torch.float64
    return torch.float32


def rbf_kernel_batched(x1: torch.Tensor, x2: torch.Tensor, length: float) -> torch.Tensor:
    dist_sq = torch.cdist(x1, x2, p=2) ** 2
    return torch.exp(-dist_sq / (2.0 * float(length) ** 2))


def rbf_kernel_matrix(x1: torch.Tensor, x2: torch.Tensor, length: float) -> torch.Tensor:
    dist_sq = torch.cdist(x1, x2, p=2) ** 2
    return torch.exp(-dist_sq / (2.0 * float(length) ** 2))


def build_basis(
    x_train: list[np.ndarray],
    basis_size: int | None,
    random_state: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_all_np = np.concatenate(x_train, axis=0)
    if basis_size is not None and basis_size < x_all_np.shape[0]:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(x_all_np.shape[0], size=int(basis_size), replace=False)
        x_all_np = x_all_np[indices]
    return torch.as_tensor(x_all_np, dtype=dtype, device=device)


def estimate_subject_weights(
    x_train: list[np.ndarray],
    y_train: list[np.ndarray],
    params: dict[str, float],
    jitter: float,
    basis_size: int | None,
    random_state: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_all = build_basis(x_train, basis_size=basis_size, random_state=random_state, device=device, dtype=dtype)
    x_train_tensor = torch.stack([torch.as_tensor(x, dtype=dtype, device=device) for x in x_train], dim=0)
    y_train_tensor = torch.stack([torch.as_tensor(y, dtype=dtype, device=device) for y in y_train], dim=0)

    kernel_matrix = rbf_kernel_matrix(x_all, x_all, params["length"])
    kernel_matrix = kernel_matrix + float(jitter) * torch.eye(x_all.shape[0], dtype=dtype, device=device)
    kernel_chol = torch.linalg.cholesky(kernel_matrix)

    beta = 1.0 / float(params["noise_level"])
    k_xb = rbf_kernel_batched(x_train_tensor, x_all.unsqueeze(0).expand(x_train_tensor.shape[0], -1, -1), params["length"])
    z = torch.cholesky_solve(k_xb.transpose(1, 2), kernel_chol.unsqueeze(0).expand(x_train_tensor.shape[0], -1, -1))
    s = torch.bmm(k_xb, z)
    m = s + float(params["noise_level"]) * torch.eye(s.shape[-1], dtype=dtype, device=device).unsqueeze(0)
    v1 = torch.bmm(z, y_train_tensor.unsqueeze(-1))
    v2 = torch.bmm(k_xb, v1)
    v3 = torch.linalg.solve(m, v2)
    weights = beta * (v1 - torch.bmm(z, v3)).squeeze(-1)
    return x_all, weights


def predict_subjects(
    x_val: list[np.ndarray],
    x_all: torch.Tensor,
    weights: torch.Tensor,
    length: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_val_tensor = torch.stack([torch.as_tensor(x, dtype=dtype, device=device) for x in x_val], dim=0)
    k_val = rbf_kernel_batched(x_val_tensor, x_all.unsqueeze(0).expand(x_val_tensor.shape[0], -1, -1), length)
    return torch.bmm(k_val, weights.unsqueeze(-1)).squeeze(-1)


def cross_validated_score(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    params: dict[str, float],
    folds: int,
    random_state: int,
    jitter: float,
    basis_size: int | None,
    device: torch.device,
    dtype: torch.dtype,
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

        x_train_fold = []
        y_train_fold = []
        x_val_fold = []
        y_val_fold = []
        for subject_index, (x_subject, y_subject) in enumerate(zip(x_list, y_list)):
            train_indices, val_indices = splitters[subject_index][fold_index]
            x_train_fold.append(x_subject[train_indices])
            y_train_fold.append(y_subject[train_indices])
            x_val_fold.append(x_subject[val_indices])
            y_val_fold.append(y_subject[val_indices])

        x_all, weights = estimate_subject_weights(
            x_train=x_train_fold,
            y_train=y_train_fold,
            params=params,
            jitter=jitter,
            basis_size=basis_size,
            random_state=random_state,
            device=device,
            dtype=dtype,
        )

        predictions = predict_subjects(
            x_val=x_val_fold,
            x_all=x_all,
            weights=weights,
            length=params["length"],
            device=device,
            dtype=dtype,
        )
        y_val_tensor = torch.stack([torch.as_tensor(y, dtype=dtype, device=device) for y in y_val_fold], dim=0)
        subject_mse = torch.mean((predictions - y_val_tensor) ** 2, dim=1)
        pred_z = zscore_tensor(predictions)
        y_z = zscore_tensor(y_val_tensor)
        subject_shape_rmse = torch.sqrt(torch.mean((pred_z - y_z) ** 2, dim=1))
        fold_shape_mean = float(subject_shape_rmse.mean().item())
        fold_mse_mean = float(subject_mse.mean().item())
        fold_shape_scores.append(fold_shape_mean)
        fold_mse_scores.append(fold_mse_mean)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats(device)
            memory_text = f", peak_gpu_mem={memory_gb:.2f} GB"
        else:
            memory_text = ""

        fold_elapsed = time.perf_counter() - fold_started_at
        print(
            f"{tag}finished fold {fold_index + 1}/{folds}: "
            f"shape_rmse={fold_shape_mean:.6f}, mean_mse={fold_mse_mean:.6f}, "
            f"elapsed={fold_elapsed/60:.1f} min{memory_text}",
            flush=True,
        )

    return {
        "mean_rmse": float(np.mean(fold_shape_scores)),
        "mean_mse": float(np.mean(fold_mse_scores)),
    }


def search_hyperparameters(
    output_dir: Path,
    folds: int,
    random_state: int,
    length_grid: np.ndarray,
    noise_grid: np.ndarray,
    jitter: float,
    basis_size: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    x_list, y_list = load_psycho_data2_2.load_data()
    print("ハイパーパラメータ探索のデータ準備が完了しました。", flush=True)
    print(f"device={device}, dtype={dtype}, basis_size={basis_size}", flush=True)

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
        print(
            f"[{index:03d}/{len(param_grid):03d}] evaluating "
            f"length={length:.6g}, noise={noise_level:.6g}",
            flush=True,
        )
        score_summary = cross_validated_score(
            x_list=x_list,
            y_list=y_list,
            params=params,
            folds=folds,
            random_state=random_state,
            jitter=jitter,
            basis_size=basis_size,
            device=device,
            dtype=dtype,
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    search_hyperparameters(
        output_dir=args.output_dir,
        folds=args.folds,
        random_state=args.random_state,
        length_grid=np.logspace(np.log10(args.length_min), np.log10(args.length_max), args.length_points),
        noise_grid=np.logspace(np.log10(args.noise_min), np.log10(args.noise_max), args.noise_points),
        jitter=args.jitter,
        basis_size=args.basis_size,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
