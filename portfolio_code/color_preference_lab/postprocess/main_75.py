import os

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from botorch.models import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

from ego.stimulation.color import image_create_monochrome_ref


def predict_mean(result, response):
    # 予測平均を出す
    point = torch.linspace(-1, 1, 256)
    p1, p2 = torch.meshgrid(point, point)
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
    x_25 = np.percentile(x, 25, interpolation="nearest")
    x_50 = np.percentile(x, 50, interpolation="nearest")
    x_75 = np.percentile(x, 75, interpolation="nearest")
    x_max = np.max(x)

    x_min_index = np.where(x == x_min)[0]
    x_25_index = np.where(x == x_25)[0]
    x_50_index = np.where(x == x_50)[0]
    x_75_index = np.where(x == x_75)[0]
    x_max_index = np.where(x == x_max)[0]

    x_percentile = np.vstack((x_min_index, x_25_index, x_50_index, x_75_index, x_max_index))

    return x_percentile


def main(name_list_file):
    # 全実験参加者の予測平均の算出
    name_list = pd.read_csv(name_list_file, header=None)
    name_list = np.array(name_list).flatten().tolist()

    points = torch.linspace(-1, 1, 256)
    p1, p2 = torch.meshgrid(points, points)
    point_mesh = torch.hstack((p2.reshape(-1, 1), p1.reshape(-1, 1)))

    point = torch.arange(256)

    for i in range(len(name_list)):

        result_dir = "./result/75/" + name_list[i] + "/"
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        # データの読み込み（result, response）
        print("Now loading:", name_list[i])
        result = torch.load("../data/" + name_list[i] + "_result.pt")
        response = torch.load("../data/" + name_list[i] + "_response.pt")
        # 各実験参加者の予測平均の算出
        mu = predict_mean(result=result[:150], response=response[:75])
        torch.save(mu.reshape(-1, 1), result_dir + name_list[i] + "_mu.pt")

        point_and_mu = torch.hstack((point_mesh, mu.reshape(-1, 1)))
        torch.save(point_and_mu, result_dir + name_list[i] + "_point_and_mu.pt")

        mu_percentile = x_percentile(mu)

        mu_percentile_point = torch.vstack((point_and_mu[mu_percentile[0]][:2],
                                            point_and_mu[mu_percentile[1]][:2],
                                            point_and_mu[mu_percentile[2]][:2],
                                            point_and_mu[mu_percentile[3]][:2],
                                            point_and_mu[mu_percentile[4]][:2]))

        torch.save(mu_percentile_point, result_dir + name_list[i] + "percentile.pt")

        print(point_and_mu[mu_percentile[0]][0][:2])
        print(point_and_mu[mu_percentile[1]][0][:2])
        print(point_and_mu[mu_percentile[2]][0][:2])
        print(point_and_mu[mu_percentile[3]][0][:2])
        print(point_and_mu[mu_percentile[4]][0][:2])

        i_path = result_dir + name_list[i] + "_min.png"
        image_create_monochrome_ref(r=point_and_mu[mu_percentile[0]][0][:2].tolist(), i_path=i_path)

        i_path = result_dir + name_list[i] + "_25.png"
        image_create_monochrome_ref(r=point_and_mu[mu_percentile[1]][0][:2].tolist(), i_path=i_path)

        i_path = result_dir + name_list[i] + "_50.png"
        image_create_monochrome_ref(r=point_and_mu[mu_percentile[2]][0][:2].tolist(), i_path=i_path)

        i_path = result_dir + name_list[i] + "_75.png"
        image_create_monochrome_ref(r=point_and_mu[mu_percentile[3]][0][:2].tolist(), i_path=i_path)

        i_path = result_dir + name_list[i] + "_max.png"
        image_create_monochrome_ref(r=point_and_mu[mu_percentile[4]][0][:2].tolist(), i_path=i_path)

        plt.figure(figsize=(12, 10))
        plt.xticks(np.arange(0, 256, 25))
        plt.yticks(np.arange(0, 256, 25))
        # 各実験参加者の予測平均の分布
        plt.contourf(point, point, mu.detach().numpy().reshape(256, 256))
        # ref.jpgのa, bの値をplot
        if session == 1:
            a_b = np.array([-0.3, -0.3])
        elif session == 2:
            a_b = np.array([-0.3, 0.3])
        elif session == 3:
            a_b = np.array([0.3, -0.3])
        elif session == 4:
            a_b = np.array([0.3, 0.3])

        a_b = ((a_b + 1) / 2) * 255
        mu_percentile_point_reverse = ((mu_percentile_point + 1) / 2) * 255
        mu_percentile_point_reverse = mu_percentile_point_reverse.to('cpu').detach().numpy().copy()
        print(mu_percentile_point_reverse[4])

        plt.plot(a_b[0], a_b[1], "o", color='red', label="Reference")
        plt.plot(mu_percentile_point_reverse[4][0],
                 mu_percentile_point_reverse[4][1],
                 "o", color="blue", label="Max")
        plt.xlabel("a*", fontsize=24)
        plt.ylabel("b*", fontsize=24)
        plt.colorbar()
        plt.legend(fontsize=24)
        plt.savefig(result_dir + "/" + name_list[i] + "_mu.pdf")


def main_all(name_list_file, session):
    # main()で出力した各実験参加者の予測平均の平均を算出

    result_dir = "./result/all/75/" + str(session) + "/"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # meshgridの作成
    point = torch.linspace(-1, 1, 256)
    p1, p2 = torch.meshgrid(point, point)
    point_mesh = torch.hstack((p2.reshape(-1, 1), p1.reshape(-1, 1)))

    name_list = pd.read_csv(name_list_file, header=None)
    name_list = np.array(name_list).flatten().tolist()

    # 1人目の予測平均のリストの読み込み
    result_list_0 = torch.load("./result/75/" + name_list[0] + "/" + name_list[0] + "_mu.pt")

    # 1人目と2人目の予測平均のリストの結合
    result_list = torch.hstack((result_list_0, torch.load("./result/75/" + name_list[1] + "/" + name_list[1] + "_mu.pt")))
    # result_listと3人目以降の予測平均のリストの結合
    for i in range(2, len(name_list)):
        result_list = torch.hstack(
            (result_list, torch.load("./result/75/" + name_list[i] + "/" + name_list[i] + "_mu.pt")))

    # ここで平均予測平均の計算と保存
    mean_result_list = result_list.mean(dim=1)
    torch.save(mean_result_list, result_dir + "_all_mu.pt")

    mu_percentile = x_percentile(mean_result_list)

    point_and_mu = torch.hstack((point_mesh, mean_result_list.reshape(-1, 1)))
    torch.save(point_and_mu, result_dir + "all_point_and_mu.pt")

    mu_percentile_point = torch.vstack((point_and_mu[mu_percentile[0]][:2],
                                        point_and_mu[mu_percentile[1]][:2],
                                        point_and_mu[mu_percentile[2]][:2],
                                        point_and_mu[mu_percentile[3]][:2],
                                        point_and_mu[mu_percentile[4]][:2]))

    torch.save(mu_percentile_point, result_dir + "all_percentile.pt")

    print(point_and_mu[mu_percentile[0]][0][:2])
    print(point_and_mu[mu_percentile[1]][0][:2])
    print(point_and_mu[mu_percentile[2]][0][:2])
    print(point_and_mu[mu_percentile[3]][0][:2])
    print(point_and_mu[mu_percentile[4]][0][:2])

    i_path = result_dir + "all_min.png"
    image_create_monochrome_ref(r=point_and_mu[mu_percentile[0]][0][:2].tolist(), i_path=i_path)

    i_path = result_dir + "all_25.png"
    image_create_monochrome_ref(r=point_and_mu[mu_percentile[1]][0][:2].tolist(), i_path=i_path)

    i_path = result_dir + "all_50.png"
    image_create_monochrome_ref(r=point_and_mu[mu_percentile[2]][0][:2].tolist(), i_path=i_path)

    i_path = result_dir + "all_75.png"
    image_create_monochrome_ref(r=point_and_mu[mu_percentile[3]][0][:2].tolist(), i_path=i_path)

    i_path = result_dir + "all_max.png"
    image_create_monochrome_ref(r=point_and_mu[mu_percentile[4]][0][:2].tolist(), i_path=i_path)

    point_and_mu = torch.hstack((point_mesh, mean_result_list.reshape(-1, 1)))
    torch.save(point_and_mu, result_dir + "all_point_and_mu.pt")

    point = torch.arange(256)
    plt.figure(figsize=(12, 10))
    plt.xticks(np.arange(0, 256, 25))
    plt.yticks(np.arange(0, 256, 25))
    # 各実験参加者の予測平均の分布
    plt.contourf(point, point, mean_result_list.detach().numpy().reshape(256, 256))
    # ref.jpgのa, bの値をplot

    if session == 1:
        a_b = np.array([-0.3, -0.3])
    elif session == 2:
        a_b = np.array([-0.3, 0.3])
    elif session == 3:
        a_b = np.array([0.3, -0.3])
    elif session == 4:
        a_b = np.array([0.3, 0.3])

    a_b = ((a_b + 1) / 2) * 255
    mu_percentile_point_reverse = ((mu_percentile_point + 1) / 2) * 255
    mu_percentile_point_reverse = mu_percentile_point_reverse.to('cpu').detach().numpy().copy()
    print(mu_percentile_point_reverse[4])
    plt.plot(a_b[0], a_b[1], "o", color='red', label="Reference")
    plt.plot(mu_percentile_point_reverse[4][0],
             mu_percentile_point_reverse[4][1],
             "o", color="blue", label="Max")
    plt.xlabel("a*", fontsize=24)
    plt.ylabel("b*", fontsize=24)
    plt.colorbar()
    plt.legend(fontsize=24)
    plt.savefig(result_dir + "all_mu.pdf")


if __name__ == '__main__':

    for session in range(1, 5):
        name_list_file = "./name_list" + str(session) + ".csv"
        main(name_list_file=name_list_file)
        main_all(name_list_file=name_list_file, session=session)
