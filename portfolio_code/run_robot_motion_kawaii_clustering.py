from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ROBOT_ROOT = PROJECT_ROOT / "robot_motion_kawaii"
if str(ROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clustering for the robot motion preference experiment.")
    parser.add_argument("--data-dir", type=Path, default=ROBOT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROBOT_ROOT / "output")
    parser.add_argument("--model-dim", type=int, default=20)
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--length", type=float, default=2.0)
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import cluster_meanfunc_maxmin as cluster_module
    import load_psycho_data2_2

    cluster_module.CONFIG["model_dim"] = args.model_dim
    cluster_module.CONFIG["num_clusters"] = args.num_clusters
    cluster_module.CONFIG["jitter"] = args.jitter
    cluster_module.CONFIG["output_dir"] = str(args.output_dir)
    cluster_module.CONFIG["fixed_params"] = {"length": args.length, "noise_level": args.noise_level}

    x_train, y_train = load_psycho_data2_2.load_data(data_dir=args.data_dir)
    n_dims = x_train[0].shape[1]
    model = cluster_module.SparseKPCA(
        x_list=x_train,
        y_list=y_train,
        params=cluster_module.CONFIG["fixed_params"],
        modelDim=cluster_module.CONFIG["model_dim"],
        jitter=cluster_module.CONFIG["jitter"],
    )
    model.fit()
    cluster_module.perform_clustering_and_save(model, n_dims)


if __name__ == "__main__":
    main()
