#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")

from kernel import InverseMultiquadricKernelPytouch
from mtgp import MultiTaskGaussianProcessImplicitSurfaces

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
# max_iter     = 20
max_iter     = 100
lr           = 0.0001
sigma        = torch.tensor(-5.168)

plot = True
# plot = False
save_data = True
# save_data = False
save_movie = True
save_movie = False

file_path = "../data/high_high/mtgpis"




def judge_square(po):
    if (0.1685 < po[0] < 0.4315) and (0.1685 < po[1] < 0.4315):
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
    current_po = np.array([0.3, 0.5])
    po_list    = np.array([current_po])
    var_list   = []
    simirality_list = []
    times      = []
    kernel     = InverseMultiquadricKernelPytouch([kernel_param])

    # visual data
    N1 = 10
    x1_0 = (np.concatenate([np.linspace(15, 45, N1), np.ones(N1)*45, np.linspace(45, 15, N1), np.ones(N1)*15]) ) * rate
    x1_1 = (np.concatenate([np.linspace(13, 47, N1), np.ones(N1)*47, np.linspace(47, 13, N1), np.ones(N1)*13]) ) * rate
    x1   = np.concatenate([x1_0, x1_1])[:, None]

    y1_0 = (np.concatenate([np.ones(N1)*15, np.linspace(15, 45, N1), np.ones(N1)*45, np.linspace(45, 15, N1)]) ) * rate
    y1_1 = (np.concatenate([np.ones(N1)*13, np.linspace(13, 47, N1), np.ones(N1)*47, np.linspace(47, 13, N1)]) ) * rate
    y1   = np.concatenate([y1_0, y1_1])[:, None]

    X1 = np.concatenate([x1,y1],1)
    Y1 = np.concatenate([np.zeros(len(x1_0)), np.ones(len(x1_1)) ])[:, None]
    T1 = 0

    # true object
    N2 = 10
    x2_0 = (np.concatenate([np.linspace(17, 43, N2), np.ones(N2)*43, np.linspace(43, 17, N2), np.ones(N2)*17]) ) * rate
    y2_0 = (np.concatenate([np.ones(N2)*17, np.linspace(17, 43, N2), np.ones(N2)*43, np.linspace(43, 17, N2)]) ) * rate

    # test data
    x   = np.linspace(0, 60, 800)[:, None] * rate
    y   = np.linspace(0, 60, 800)[:, None] * rate
    # x   = np.linspace(-20, 80, 200)[:, None] * rate
    # y   = np.linspace(-20, 80, 200)[:, None] * rate
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
            outer = pat.Rectangle(xy = (0.15, 0.15), width = 0.3, height = 0.3,linewidth=2, linestyle='dashdot', ec=color_cycle[5], fill=False)
            inner = pat.Rectangle(xy = (0.17, 0.17), width = 0.26, height = 0.26, color=color_cycle[5])
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
    X2 = np.append(X2, [current_po + np.array([-0.02, 0.02])], axis=0)
    Y2 = np.append(Y2, [1])[:,None]
    T2 = 1


    for i in range(105):
        s = time.time()
        print("=========================================================================")
        print("STEP: {}".format(i))

        if i == 0:
            task_kernel_params = torch.tensor([[-0.2881, -0.3729], [0.27, -1.354]])
        else:
            task_kernel_params = gp_model.task_kernel_params


        print("taks kernel:", task_kernel_params)
        X1_t = torch.from_numpy(X1).float()
        X2_t = torch.from_numpy(X2).float()
        Y1_t = torch.from_numpy(Y1).float()
        Y2_t = torch.from_numpy(Y2).float()

        gp_model = MultiTaskGaussianProcessImplicitSurfaces([X1_t, X2_t], [Y1_t, Y2_t], [T1,T2], kernel, task_kernel_params=task_kernel_params, sigma=sigma)
        gp_model.learning(max_iter=max_iter, lr=lr)


        # save model
        with open('{}/model/mtgp_model_{}.pickle'.format(file_path, movie_num), mode='wb') as fp:
            pickle.dump(gp_model, fp)

        mm2, ss2  = gp_model.predict(XX, T2)
        # mean_zero = np.where(abs(mm2.T[0]) < 0.05)
        mean_zero = np.where(abs(mm2.T[0]) < 0.03)
        surf_x    = xx.T[0][mean_zero]
        surf_y    = yy.T[0][mean_zero]
        var       = np.array(ss2.T[0][mean_zero])

        var_list.append(np.mean(var))
        simirality_list.append(gp_model.task_params_to_psd().numpy())

        ########################################## Plot ###################################################################
        if plot:
            ax = fig.add_subplot(111)

            ss2 = ss2.reshape(x.shape)
            z   = ss2.numpy()
            xyz = plt.pcolormesh(x, y, z, cmap=cm.Purples, vmax=z.max(), vmin=z.min())

            outer = pat.Rectangle(xy = (0.15, 0.15), width = 0.3, height = 0.3,linewidth=2, linestyle='dashdot', ec=color_cycle[5], fill=False)
            inner = pat.Rectangle(xy = (0.17, 0.17), width = 0.26, height = 0.26, color=color_cycle[5])
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

            # set_plt()
            # plt.scatter(surf_x, surf_y, s=3, color=(1.0, 0.0, 0.0), zorder=8)
            # plt.savefig('{}/estimate_surface/step{}.png'.format(file_path, i+1), dpi=300, bbox_inches="tight", pad_inches=0.05)
            # plt.clf()

            set_plt()
            plt.plot(x2_0, y2_0, linewidth=3, color=(1.0, 0.0, 0.0))
            plt.savefig('{}/true_surface/step{}.png'.format(file_path, i+1), dpi=300, bbox_inches="tight", pad_inches=0.05)
            plt.clf()

        _, direrction = gp_model.predict_direction(torch.Tensor(current_po[:,None].T), T2)
        direrction    = direrction.numpy()
        current_po   += alpha * direrction.T[0]

        normal, _ = gp_model.predict_direction(torch.Tensor(current_po[:,None].T), T2)
        normal    = normal.numpy()
        # print("direction:", direrction)
        # print("normal:", normal)

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

                if n % 10 == 0:
                    if judge_square(current_po + normal.T[0] * rate * 2) == 1:
                        X2 = np.append(X2, [current_po + normal.T[0] * rate * 2], axis=0)
                        Y2 = np.append(Y2, [1])[:,None]

                    X1_t = torch.from_numpy(X1).float()
                    X2_t = torch.from_numpy(X2).float()
                    Y1_t = torch.from_numpy(Y1).float()
                    Y2_t = torch.from_numpy(Y2).float()

                    task_kernel_params = gp_model.task_kernel_params
                    gp_model = MultiTaskGaussianProcessImplicitSurfaces([X1_t, X2_t], [Y1_t, Y2_t], [T1,T2], kernel, task_kernel_params=task_kernel_params, sigma=sigma)
                    gp_model.learning(max_iter=20, lr=lr)

                    normal, direrction = gp_model.predict_direction(torch.Tensor(current_po[:,None].T), T2)
                    normal     = normal.numpy()

                current_po -= 0.0001 * normal.T[0]
                n += 1

        normal, _ = gp_model.predict_direction(torch.Tensor(current_po[:,None].T), T2)
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
    print("simirality:", simirality_list)

    if save_data:
        np.save('{}/value/path'.format(file_path), po_list)
        np.save('{}/value/var'.format(file_path), var_list)
        np.save('{}/value/simirality'.format(file_path), simirality_list)


    plt.plot(np.arange(var_list.shape[0]), var_list)
    plt.show()

