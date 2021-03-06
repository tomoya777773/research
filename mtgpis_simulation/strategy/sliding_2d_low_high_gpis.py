#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")

from kernel import InverseMultiquadricKernelPytouch
from mtgp import GaussianProcessImplicitSurfaces

import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import pickle

plt.style.use('ggplot')
color_cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# hyper parameter
alpha        = 0.01
kernel_param = 0.1
rate         = 0.01
sigma        = torch.tensor(-5.168)


plot = True
# plot = False
save_data = True
# save_data = False
save_movie = True
# save_movie = False

file_path = "../data/low_high/gpis"


def judge_square(po):
    if (0.118 < po[0] < 0.282) and (0.118 < po[1] < 0.382):
        return -1
    else:
        return 1

def set_plt():
    plt.xlim(0, 0.6)
    plt.ylim(0, 0.6)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.axes().set_aspect('equal', 'datalim')


if __name__=="__main__":
    current_po = np.array([0.25, 0.55])
    po_list    = np.array([current_po])
    var_list   = []
    times      = []
    kernel     = InverseMultiquadricKernelPytouch([kernel_param])

    # visual data
    N1 = 10
    x1_0 = (np.concatenate([np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1), np.ones(N1)*10]) ) * rate
    y1_0 = (np.concatenate([np.ones(N1)*10, np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1)]) ) * rate

    # true object
    N2 = 10
    x2_0 = (np.concatenate([np.linspace(12, 28, N2), np.ones(N2)*28, np.linspace(28, 12, N2), np.ones(N2)*12]) ) * rate
    y2_0 = (np.concatenate([np.ones(N2)*12, np.linspace(12, 38, N2), np.ones(N2)*38, np.linspace(38, 12, N2)]) ) * rate

    # test data
    x   = np.linspace(0, 60, 800)[:, None] * rate
    y   = np.linspace(0, 60, 800)[:, None] * rate
    # x   = np.linspace(0, 60, 200)[:, None] * rate
    # y   = np.linspace(0, 60, 200)[:, None] * rate
    x,y = np.meshgrid(x, y)
    xx  = x.ravel()[:, None]
    yy  = y.ravel()[:, None]
    XX  = np.concatenate([xx, yy], 1)
    XX  = torch.from_numpy(XX).float()

    fig = plt.figure(figsize=(3.0, 3.0), dpi=300)

    movie_num = 1
    while True:
        if judge_square(current_po - np.array([0, 0.002])) == -1:
            break
        current_po[1] -= 0.002
        po_list  = np.append(po_list, [current_po], axis = 0)
        ########################################## Plot ###################################################################
        if plot:
            ax = fig.add_subplot(111)

            back  = pat.Rectangle(xy = (0, 0), width = 0.6, height = 0.6 , color="white")
            outer = pat.Rectangle(xy = (0.1, 0.1), width = 0.4, height = 0.4,linewidth=2, linestyle='dashdot', ec=color_cycle[5], fill=False)
            inner = pat.Rectangle(xy = (0.12, 0.12), width = 0.16, height = 0.26, color=color_cycle[5])
            ax.add_patch(back)
            ax.add_patch(outer)
            ax.add_patch(inner)

            plt.plot(po_list[:, 0], po_list[:, 1], '--', color='black', linewidth=2, zorder=9)
            plt.scatter(po_list[-1, 0], po_list[-1, 1], c='black', s=20, marker="o",zorder=10)

            set_plt()
            if save_movie:
                plt.savefig('{}/movie/step{}.png'.format(file_path, movie_num), dpi=300, pad_inches=0.05)
                movie_num += 1
            plt.draw()
            plt.pause(0.001)
            plt.clf()
        ########################################## Plot ###################################################################

    # tactile data
    X2 = np.array([current_po])
    Y2 = np.array([[0]])
    X2 = np.append(X2, [current_po + np.array([-0.01, 0.02])], axis=0)
    Y2 = np.append(Y2, [1])[:,None]
    X2 = np.append(X2, [current_po + np.array([0.01, 0.02])], axis=0)
    Y2 = np.append(Y2, [1])[:,None]

    kernel = InverseMultiquadricKernelPytouch([kernel_param])

    for i in range(85):
        s = time.time()
        print("=========================================================================")
        print("STEP: {}".format(i))

        X2_t = torch.from_numpy(X2).float()
        Y2_t = torch.from_numpy(Y2).float()

        gp_model = GaussianProcessImplicitSurfaces(X2_t, Y2_t, kernel, sigma=sigma)

        # save model
        with open('{}/model/gp_model_{}.pickle'.format(file_path, movie_num), mode='wb') as fp:
            pickle.dump(gp_model, fp)


        mm2, ss2  = gp_model.predict(XX)
        mean_zero = np.where(abs(mm2.T[0]) < 0.03)
        surf_x    = xx.T[0][mean_zero]
        surf_y    = yy.T[0][mean_zero]
        var       = np.array(ss2.T[0][mean_zero])

        var_list.append(np.mean(var))

        ########################################## Plot ###################################################################
        if plot:
            ax = fig.add_subplot(111)

            ss2 = ss2.reshape(x.shape)
            z   = ss2.numpy()
            xyz = plt.pcolormesh(x, y, z, cmap=cm.Purples, vmax=z.max(), vmin=z.min())

            outer = pat.Rectangle(xy = (0.1, 0.1), width = 0.4, height = 0.4,linewidth=2, linestyle='dashdot', ec=color_cycle[5], fill=False)
            inner = pat.Rectangle(xy = (0.12, 0.12), width = 0.16, height = 0.26, color=color_cycle[5])
            ax.add_patch(outer)
            ax.add_patch(inner)

            plt.plot(po_list[:, 0], po_list[:, 1], '--', color='black', linewidth=2, zorder=9)
            plt.scatter(po_list[-1, 0], po_list[-1, 1], c='black', s=20, marker="o",zorder=10)
            plt.scatter(surf_x, surf_y, s=3, c=color_cycle[0], zorder=8)
            set_plt()

            if save_movie:
                plt.savefig('{}/movie/step{}.png'.format(file_path, movie_num), dpi=300, pad_inches=0.05)
                movie_num += 1
            plt.draw()
            plt.pause(0.001)
            plt.clf()
        ########################################## Plot ###################################################################

        if save_data:
            ax = fig.add_subplot(111)
            back  = pat.Rectangle(xy = (0, 0), width = 0.6, height = 0.6 , color="white")
            ax.add_patch(back)

            set_plt()
            plt.scatter(surf_x, surf_y, s=3, color=(1.0, 0.0, 0.0), zorder=8)
            plt.savefig('{}/estimate_surface/step{}.png'.format(file_path, i+1), dpi=300, bbox_inches="tight", pad_inches=0.05)
            plt.clf()

            # set_plt()
            # plt.plot(x2_0, y2_0, linewidth=3, color=(1.0, 0.0, 0.0))
            # plt.savefig('{}/true_surface/step{}.png'.format(file_path, i+1), dpi=300, bbox_inches="tight", pad_inches=0.05)
            # plt.clf()

        _, direrction = gp_model.predict_direction(torch.Tensor(current_po[:,None].T))
        direrction    = direrction.numpy()
        current_po   += alpha * direrction.T[0]

        normal, _ = gp_model.predict_direction(torch.Tensor(current_po[:,None].T))
        normal    = normal.numpy()
        print("direction:", direrction)
        print("normal:", normal)

        judge = judge_square(current_po)

        if judge == -1: # in
            while True:
                if judge_square(current_po) == 1:
                    X2 = np.append(X2, [current_po], axis=0)
                    Y2 = np.append(Y2, [0])[:,None]
                    po_list = np.append(po_list, [current_po], axis = 0)
                    break
                current_po += 0.0001 * normal.T[0]

        elif judge == 1: # out
            n = 1
            while True:
                if judge_square(current_po - 0.0001 * normal.T[0]) == -1:
                    X2 = np.append(X2, [current_po], axis=0)
                    Y2 = np.append(Y2, [0])[:,None]
                    po_list = np.append(po_list, [current_po], axis = 0)
                    break

                if n % 5 == 0:
                    if judge_square(current_po + normal.T[0] * rate * 2) == 1:
                        X2 = np.append(X2, [current_po + normal.T[0] * rate * 2], axis=0)
                        Y2 = np.append(Y2, [1])[:,None]

                    X2_t = torch.from_numpy(X2).float()
                    Y2_t = torch.from_numpy(Y2).float()

                    gp_model = GaussianProcessImplicitSurfaces(X2_t, Y2_t, kernel, sigma=sigma)

                    normal, direrction = gp_model.predict_direction(torch.Tensor(current_po[:,None].T))
                    normal     = normal.numpy()

                current_po -= 0.0001 * normal.T[0]
                n += 1

        normal, _ = gp_model.predict_direction(torch.Tensor(current_po[:,None].T))
        normal    = normal.numpy()
        if judge_square(current_po + normal.T[0] * rate * 2) == 1:
            X2 = np.append(X2, [current_po + normal.T[0] * rate * 2], axis=0)
            Y2 = np.append(Y2, [1])[:,None]

    #     print("time:", time.time() - s)
    #     times.append(time.time() - s)
    # print("times:", sum(times)/ len(times))

    po_list  = np.array(po_list)
    var_list = np.array(var_list)
    print("po_list:", po_list)
    print("var_list:", var_list)

    if save_data:
        np.save('{}/value/var'.format(file_path), var_list)
        np.save('{}/value/path'.format(file_path), po_list)

    plt.plot(np.arange(var_list.shape[0]), var_list)
    plt.show()

