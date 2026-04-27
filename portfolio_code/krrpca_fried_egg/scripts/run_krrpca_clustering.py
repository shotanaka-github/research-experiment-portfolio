from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from krrpca_fried_egg import CommonBasisKernelRidgePCA, KernelRidgePCAParameters, load_fried_egg_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ward clustering on KRRPCA subject weights for fried-egg data."
    )
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--label-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/clustering"))
    parser.add_argument("--length", type=float, default=0.3)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--basis-size", type=int, default=256)
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def rkhs_distance_matrix(weights: np.ndarray, kernel_matrix: np.ndarray) -> np.ndarray:
    diff = weights[:, None, :] - weights[None, :, :]
    dist_sq = np.einsum("ijn,nm,ijm->ij", diff, kernel_matrix, diff)
    return np.sqrt(np.maximum(dist_sq, 0.0))


def find_cluster_medoids(
    distance_matrix: np.ndarray, subject_ids: np.ndarray, cluster_labels: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, int | float]] = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_dist = distance_matrix[np.ix_(indices, indices)]
        total_distance = cluster_dist.sum(axis=1)
        medoid_local_index = int(np.argmin(total_distance))
        medoid_index = int(indices[medoid_local_index])
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(indices)),
                "medoid_subject_id": int(subject_ids[medoid_index]),
                "mean_within_cluster_distance": float(cluster_dist.mean()),
            }
        )
    return pd.DataFrame(rows)


def save_dendrogram(linked: np.ndarray, subject_ids: np.ndarray, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    figure, axis = plt.subplots(figsize=(12, 6))
    dendrogram(linked, labels=[str(subject_id) for subject_id in subject_ids], ax=axis)
    axis.set_title("KRRPCA Ward Clustering")
    axis.set_xlabel("Subject ID")
    axis.set_ylabel("RKHS distance")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_latent_plot(
    latent_scores: np.ndarray, cluster_labels: np.ndarray, subject_ids: np.ndarray, output_path: Path
) -> None:
    if latent_scores.shape[1] < 2:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    figure, axis = plt.subplots(figsize=(8, 8))
    scatter = axis.scatter(
        latent_scores[:, 0],
        latent_scores[:, 1],
        c=cluster_labels,
        cmap="tab10",
        s=50,
    )
    for subject_id, x_coord, y_coord in zip(subject_ids, latent_scores[:, 0], latent_scores[:, 1]):
        axis.text(x_coord, y_coord, str(subject_id), fontsize=8, ha="left", va="bottom")
    axis.set_xlabel("Latent dimension 1")
    axis.set_ylabel("Latent dimension 2")
    axis.set_title("KRRPCA latent space colored by cluster")
    figure.colorbar(scatter, ax=axis, label="Cluster")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


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

    distance_matrix = rkhs_distance_matrix(model.weights, model.kernel_matrix)
    linked = linkage(squareform(distance_matrix, checks=False), method="ward")
    cluster_labels = fcluster(linked, t=args.num_clusters, criterion="maxclust")

    cluster_df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "cluster_id": cluster_labels,
        }
    ).sort_values(["cluster_id", "subject_id"])
    cluster_df.to_csv(args.output_dir / "cluster_membership.csv", index=False)

    distance_df = pd.DataFrame(distance_matrix, index=subject_ids, columns=subject_ids)
    distance_df.to_csv(args.output_dir / "rkhs_distance_matrix.csv")

    medoid_df = find_cluster_medoids(distance_matrix, subject_ids, cluster_labels)
    medoid_df.to_csv(args.output_dir / "cluster_summary.csv", index=False)

    if model.latent_scores is not None:
        latent_df = pd.DataFrame(
            model.latent_scores,
            columns=[f"z{i + 1}" for i in range(model.latent_scores.shape[1])],
        )
        latent_df.insert(0, "subject_id", subject_ids)
        latent_df["cluster_id"] = cluster_labels
        latent_df.to_csv(args.output_dir / "latent_coordinates_with_clusters.csv", index=False)

    save_dendrogram(linked, subject_ids, args.output_dir / "cluster_dendrogram.pdf")
    if model.latent_scores is not None:
        save_latent_plot(
            model.latent_scores,
            cluster_labels,
            subject_ids,
            args.output_dir / "latent_space_by_cluster.pdf",
        )


if __name__ == "__main__":
    main()
