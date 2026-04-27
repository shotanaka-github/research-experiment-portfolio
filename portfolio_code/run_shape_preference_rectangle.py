from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SHAPE_ROOT = PROJECT_ROOT / "shape_preference_rectangle"
if str(SHAPE_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the rectangle shape preference experiment pipeline.")
    parser.add_argument("--data-dir", type=Path, default=SHAPE_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=SHAPE_ROOT / "output")
    parser.add_argument("--num-people", type=int, default=21)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--model-dim", type=int, default=2)
    parser.add_argument("--jitter", type=float, default=1e-4)
    parser.add_argument("--length-min-exp", type=float, default=-1.0)
    parser.add_argument("--length-max-exp", type=float, default=0.0)
    parser.add_argument("--length-points", type=int, default=30)
    parser.add_argument("--noise-min-exp", type=float, default=-2.0)
    parser.add_argument("--noise-max-exp", type=float, default=0.0)
    parser.add_argument("--noise-points", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np

    import load_psycho_data
    import rectangle as rectangle_module

    config = {
        "model_dim": args.model_dim,
        "jitter": args.jitter,
        "output_dir": str(args.output_dir),
        "num_people": args.num_people,
    }
    param_grid = {
        "length": np.logspace(args.length_min_exp, args.length_max_exp, args.length_points),
        "noise_level": np.logspace(args.noise_min_exp, args.noise_max_exp, args.noise_points),
    }

    x_train, y_train = load_psycho_data.load_data(data_dir=args.data_dir, num_people=args.num_people)
    best_params, cv_results = rectangle_module.grid_search_cv(
        x_train,
        y_train,
        param_grid,
        cv=args.cv,
        model_dim=config["model_dim"],
        jitter=config["jitter"],
    )
    if cv_results:
        rectangle_module.process_and_visualize_results(cv_results, config, x_train, y_train)
        if best_params:
            rectangle_module.train_and_save_final_model(best_params, config, x_train, y_train)


if __name__ == "__main__":
    main()
