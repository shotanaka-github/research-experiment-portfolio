from os import path, remove

import numpy as np

import torch
import cv2

from botorch.models import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, qNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf


def random_gallery(n, dim):
    # ランダムなパラメータの生成
    tmp_gallery = torch.rand(n, dim, dtype=torch.float64) * 2 - 1
    tmp_gallery = tmp_gallery.tolist()
    return tmp_gallery


def observation_max_points(results, responses, bounds):
    # これまでの探索点の予測平均最大値に対応するパラメータの生成
    results = torch.Tensor(results).reshape(-1, len(bounds[0]))
    responses = torch.IntTensor(responses).reshape(-1, 2)
    model = PairwiseGP(results, responses)
    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    mll = fit_gpytorch_model(mll)

    observation_point = model.posterior(results, observation_noise=True).mean.tolist()
    next_x_index = observation_point.index(max(observation_point))
    next_x = results[next_x_index]

    return next_x


def acquisition_gallery(results, responses, bounds, q=1, num_restarts=5, num_raw_samples=20, acq_name=None):
    # 獲得関数によるパラメータの生成
    results = torch.Tensor(results).reshape(-1, len(bounds[0]))
    responses = torch.IntTensor(responses).reshape(-1, 2)
    model = PairwiseGP(results, responses)
    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    mll = fit_gpytorch_model(mll)

    if acq_name == "qNEI":
        acq_func = qNoisyExpectedImprovement(model, X_baseline=results)
    elif acq_name == "EI":
        acq_func = ExpectedImprovement(model, best_f=1.0)
    elif acq_name == "UCB":
        beta = np.sqrt(np.log(len(results) / 2) / (len(results) / 2))
        acq_func = UpperConfidenceBound(model, beta=1.0)
    else:
        acq_func = ExpectedImprovement(model, best_f=1.0)

    next_x, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=num_raw_samples
    )

    if acq_name == "qNEI":
        acq_galleries = next_x.tolist()
    else:
        prev_result_max = observation_max_points(results, responses, bounds)
        acq_galleries = torch.stack((next_x.flatten(), prev_result_max.flatten()), dim=0)

    return acq_galleries


def max_gallery(results, responses, bounds, q=1, num_restarts=5, num_raw_samples=20):
    # 実験の最後に出力するやつ
    model = PairwiseGP(results, responses)
    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    mll = fit_gpytorch_model(mll)

    acq_func = UpperConfidenceBound(model, beta=0.000001, maximize=True)

    next_x, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=num_raw_samples
    )

    tmp_max_gallery = next_x.flatten()
    tmp_max_gallery = tmp_max_gallery.tolist()
    return tmp_max_gallery


def min_gallery(results, responses, bounds, q=1, num_restarts=5, num_raw_samples=20):
    # 実験の最後に出力するやつ
    model = PairwiseGP(results, responses)
    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    mll = fit_gpytorch_model(mll)

    acq_func = UpperConfidenceBound(model, beta=0.000001, maximize=False)

    next_x, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=num_raw_samples
    )

    tmp_min_gallery = next_x.flatten()
    tmp_min_gallery = tmp_min_gallery.tolist()
    return tmp_min_gallery


def create_monochromatic_img(color, size):
    # 単色画像生成のために全て同じ数値にする
    # L, a, bをconcatenateする
    l = color[0] * np.ones((size[1], size[0], 1), dtype=np.uint8)
    a = color[1] * np.ones((size[1], size[0], 1), dtype=np.uint8)
    b = color[2] * np.ones((size[1], size[0], 1), dtype=np.uint8)
    return np.concatenate([l, a, b], axis=2)


def image_create_monochrome(i_path, r, results):
    # 単色画像生成をする関数
    if path.isfile(i_path) is True:
        remove(i_path)

    results.append(r)
    # Lの値は固定
    # a,bの値をOpenCVで扱う値の範囲に戻す
    # OpenCVでは整数しか扱えないのでintでキャスト
    l = 127
    a = ((r[0] + 1) / 2) * 255
    b = ((r[1] + 1) / 2) * 255
    l, a, b = int(l), int(a), int(b)

    color = [l, a, b]  # [l,a,b]
    size = [800, 800]  # [height,width]
    img = create_monochromatic_img(color, size)
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    cv2.imwrite(i_path, img)


def image_create_monochrome_ref(i_path, r):
    # 参考画像(実験画面の真ん中に出るやつ)を生成する関数．上の関数と中身は同じ
    if path.isfile(i_path) is True:
        remove(i_path)

    l = 127
    a = ((r[0] + 1) / 2) * 255
    b = ((r[1] + 1) / 2) * 255
    l, a, b = int(l), int(a), int(b)

    color = [l, a, b]  # [l,a,b]
    size = [800, 800]  # [height,width]
    img = create_monochromatic_img(color, size)
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    cv2.imwrite(i_path, img)


if __name__ == '__main__':
    REF_IMAGE = "./ref.jpg"
    PARAMS_ab = [-0.3, -0.3]  # session1
    # PARAMS_ab = [-0.3, 0.3]  # session2
    # PARAMS_ab = [0.3, -0.3]  # session3
    # PARAMS_ab = [0.3, 0.3]  # session4
    image_create_monochrome_ref(i_path=REF_IMAGE, r=PARAMS_ab)
