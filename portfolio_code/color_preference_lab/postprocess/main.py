from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from botorch.models import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego.stimulation.color import image_create_monochrome_ref


def predict_mean(result, response):
    # 予測平均を出す
    point = torch.linspace(-1, 1, 256)
    p1, p2 = torch.meshgrid(point, point, indexing="ij")
    point_mesh = torch.stack((p2.reshape(-1, 1), p1.reshape(-1, 1)), dim=-1)

    model = PairwiseGP(result, response)
    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    mll = fit_gpytorch_model(mll)

    mu = model.posterior(point_mesh).mean.squeeze()
    return mu.flatten()


def x_percentile(x):
    x = x.to('cpu').detach().numpy().copy()
    x = x.flatten()
    x_min = np.min(x)
    x_25 = np.percentile(x, 25, method="nearest")
    x_50 = np.percentile(x, 50, method="nearest")
    x_75 = np.percentile(x, 75, method="nearest")
    x_max = np.max(x)

    x_min_index = np.where(x == x_min)
    x_25_index = np.where(x == x_25)
    x_50_index = np.where(x == x_50)
    x_75_index = np.where(x == x_75)
    x_max_index = np.where(x == x_max)

    x_percentile = np.vstack((x_min_index, x_25_index, x_50_index, x_75_index, x_max_index))

    return x_percentile


def load_name_list(name_list_file: Path) -> list[str]:
    return pd.read_csv(name_list_file, header=None).iloc[:, 0].astype(str).tolist()


def save_percentile_images(result_dir: Path, stem: str, point_and_mu: torch.Tensor, mu_percentile: np.ndarray) -> None:
    suffixes = ["min", "25", "50", "75", "max"]
    for percentile_index, suffix in enumerate(suffixes):
        image_path = result_dir / f"{stem}_{suffix}.png"
        image_create_monochrome_ref(
            r=point_and_mu[mu_percentile[percentile_index]][0][:2].tolist(),
            i_path=str(image_path),
        )


def reference_ab(session: int) -> np.ndarray:
    mapping = {
        1: np.array([-0.3, -0.3]),
        2: np.array([-0.3, 0.3]),
        3: np.array([0.3, -0.3]),
        4: np.array([0.3, 0.3]),
    }
    return mapping[session]


def main(name_list_file: Path, session: int, data_dir: Path, output_dir: Path) -> None:
    name_list = load_name_list(name_list_file)

    point = torch.linspace(-1, 1, 256)
    p1, p2 = torch.meshgrid(point, point, indexing="ij")
    point_mesh = torch.hstack((p2.reshape(-1, 1), p1.reshape(-1, 1)))

    point = torch.arange(256)

    for name in name_list:
        result_dir = output_dir / name
        result_dir.mkdir(parents=True, exist_ok=True)

        # データの読み込み（result, response）
        print("Now loading:", name)
        result = torch.load(data_dir / f"{name}_result.pt")
        response = torch.load(data_dir / f"{name}_response.pt")
        # 各実験参加者の予測平均の算出
        mu = predict_mean(result=result, response=response)
        torch.save(mu.reshape(-1, 1), result_dir / f"{name}_mu.pt")

        point_and_mu = torch.hstack((point_mesh, mu.reshape(-1, 1)))
        torch.save(point_and_mu, result_dir / f"{name}_point_and_mu.pt")

        mu_percentile = x_percentile(mu)

        mu_percentile_point = torch.vstack((point_and_mu[mu_percentile[0]][:2],
                                            point_and_mu[mu_percentile[1]][:2],
                                            point_and_mu[mu_percentile[2]][:2],
                                            point_and_mu[mu_percentile[3]][:2],
                                            point_and_mu[mu_percentile[4]][:2]))

        torch.save(mu_percentile_point, result_dir / f"{name}_percentile.pt")

        save_percentile_images(result_dir, name, point_and_mu, mu_percentile)

        plt.figure(figsize=(10, 10))
        plt.xticks(np.arange(0, 256, 25))
        plt.yticks(np.arange(0, 256, 25))
        # 各実験参加者の予測平均の分布
        plt.contourf(point, point, mu.detach().numpy().reshape(256, 256))
        # ref.jpgのa, bの値をplot
        a_b = reference_ab(session)
        a_b = ((a_b + 1) / 2) * 255
        plt.plot(a_b[0], a_b[1], "o", color='black')
        plt.xlabel("a", fontsize=24)
        plt.ylabel("b", fontsize=24)
        plt.savefig(result_dir / f"{name}_mu.pdf")
        plt.close()


def main_all(name_list_file: Path, session: int, output_dir: Path) -> None:
    # main()で出力した各実験参加者の予測平均の平均を算出

    result_dir = output_dir / "all" / str(session)
    result_dir.mkdir(parents=True, exist_ok=True)

    # meshgridの作成
    point = torch.linspace(-1, 1, 256)
    p1, p2 = torch.meshgrid(point, point, indexing="ij")
    point_mesh = torch.hstack((p2.reshape(-1, 1), p1.reshape(-1, 1)))

    name_list = load_name_list(name_list_file)

    # 1人目の予測平均のリストの読み込み
    result_list_0 = torch.load(output_dir / name_list[0] / f"{name_list[0]}_mu.pt")

    # 1人目と2人目の予測平均のリストの結合
    result_list = torch.hstack(
        (result_list_0, torch.load(output_dir / name_list[1] / f"{name_list[1]}_mu.pt"))
    )
    # result_listと3人目以降の予測平均のリストの結合
    for i in range(2, len(name_list)):
        result_list = torch.hstack((result_list, torch.load(output_dir / name_list[i] / f"{name_list[i]}_mu.pt")))

    # ここで平均予測平均の計算と保存
    mean_result_list = result_list.mean(dim=1)
    torch.save(mean_result_list, result_dir / "all_mu.pt")

    mu_percentile = x_percentile(mean_result_list)

    point_and_mu = torch.hstack((point_mesh, mean_result_list.reshape(-1, 1)))
    torch.save(point_and_mu, result_dir / "all_point_and_mu.pt")

    mu_percentile_point = torch.vstack((point_and_mu[mu_percentile[0]][:2],
                                        point_and_mu[mu_percentile[1]][:2],
                                        point_and_mu[mu_percentile[2]][:2],
                                        point_and_mu[mu_percentile[3]][:2],
                                        point_and_mu[mu_percentile[4]][:2]))

    torch.save(mu_percentile_point, result_dir / "all_percentile.pt")
    save_percentile_images(result_dir, "all", point_and_mu, mu_percentile)

    point_and_mu = torch.hstack((point_mesh, mean_result_list.reshape(-1, 1)))
    torch.save(point_and_mu, result_dir / "all_point_and_mu.pt")

    point = torch.arange(256)
    plt.figure(figsize=(10, 10))
    plt.xticks(np.arange(0, 256, 25))
    plt.yticks(np.arange(0, 256, 25))
    # 各実験参加者の予測平均の分布
    plt.contourf(point, point, mean_result_list.detach().numpy().reshape(256, 256))
    # ref.jpgのa, bの値をplot

    a_b = reference_ab(session)
    a_b = ((a_b + 1) / 2) * 255
    plt.plot(a_b[0], a_b[1], "o", color='black')
    plt.xlabel("a", fontsize=24)
    plt.ylabel("b", fontsize=24)
    plt.savefig(result_dir / "all_mu.pdf")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess the color preference experiment.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--names-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "result")
    parser.add_argument("--sessions", nargs="*", type=int, default=[1, 2, 3, 4])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for session in args.sessions:
        name_list_file = args.names_dir / f"name_list{session}.csv"
        main(name_list_file=name_list_file, session=session, data_dir=args.data_dir, output_dir=args.output_dir)
        main_all(name_list_file=name_list_file, session=session, output_dir=args.output_dir)
