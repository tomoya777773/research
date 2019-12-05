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
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# plt.style.use("ggplot")
from scipy.spatial import ConvexHull

# hyper parameter
alpha        = 0.01
kernel_param = 0.1
rate         = 0.01
max_iter     = 1000
lr           = 0.0001
sigma        = torch.tensor(-5.168)


plot = True
# plot = False
save_data = True
# save_data = False
save_movie = True
# save_movie = False

def judge_square(po):
    if (0.22 < po[0] < 0.38) and (0.3 < po[1] < 0.48):
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

    # visual data
    N1 = 10
    x1_0 = (np.concatenate([np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1), np.ones(N1)*10]) ) * rate
    x1_1 = (np.concatenate([np.linspace(8, 52, N1), np.ones(N1)*52, np.linspace(52, 8, N1), np.ones(N1)*8]) ) * rate
    x1   = np.concatenate([x1_0, x1_1])[:, None]

    y1_0 = (np.concatenate([np.ones(N1)*10, np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1)]) ) * rate
    y1_1 = (np.concatenate([np.ones(N1)*8, np.linspace(8, 52, N1), np.ones(N1)*52, np.linspace(52, 8, N1)]) ) * rate
    y1   = np.concatenate([y1_0, y1_1])[:, None]

    X1 = np.concatenate([x1,y1],1)
    Y1 = np.concatenate([np.zeros(len(x1_0)), np.ones(len(x1_1)) ])[:, None]
    T1 = 0

    # true object
    N2 = 10
    x2_0 = (np.concatenate([np.linspace(22, 38, N2), np.ones(N2)*38, np.linspace(38, 22, N2), np.ones(N2)*22]) ) * rate
    y2_0 = (np.concatenate([np.ones(N2)*30, np.linspace(30, 48, N2), np.ones(N2)*48, np.linspace(48, 30, N2)]) ) * rate

    # test data
    x   = np.linspace(0, 60, 400)[:, None] * rate
    y   = np.linspace(0, 60, 400)[:, None] * rate
    # x   = np.linspace(0, 60, 200)[:, None] * rate
    # y   = np.linspace(0, 60, 200)[:, None] * rate
    x,y = np.meshgrid(x, y)
    xx  = x.ravel()[:, None]
    yy  = y.ravel()[:, None]
    XX  = np.concatenate([xx, yy], 1)
    XX  = torch.from_numpy(XX).float()

    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(111)


    movie_num = 1
    while True:
        if judge_square(current_po - np.array([0, 0.002])) == -1:
            break
        current_po[1] -= 0.002
        po_list  = np.append(po_list, [current_po], axis = 0)
        ########################################## Plot ###################################################################
        if plot:
            plt.plot(x1_0, y1_0, linewidth=3, c="red",zorder=6)
            plt.plot(x2_0, y2_0, linewidth=3, c="lightskyblue",zorder=7)
            plt.plot(po_list[:, 0], po_list[:, 1], linewidth=5, c='orange', zorder=9)
            plt.scatter(po_list[-1, 0], po_list[-1, 1], c="darkorange", s=30, marker="o",zorder=10)
            set_plt()
            if save_movie:
                plt.savefig('../data/high_low/mtgpis/movie/step{}.png'.format(movie_num))
                movie_num += 1
            plt.draw()
            plt.pause(0.001)
            plt.clf()
        ########################################## Plot ###################################################################

    # tactile data
    X2 = np.array([current_po])
    Y2 = np.array([[0]])
    X2 = np.append(X2, [current_po + np.array([0, 0.02])], axis=0)
    Y2 = np.append(Y2, [1])[:,None]
    T2 = 1

    kernel = InverseMultiquadricKernelPytouch([kernel_param])

    var_list = []
    simirality_list = []
    for i in range(65):
        print("=========================================================================")
        print("STEP: {}".format(i))

        if i == 0:
            task_kernel_params = torch.tensor([[-0.24897, -0.3497], [0.27, -1.3891]])
            # task_kernel_params = torch.tensor([[-0.24897, -0.8731], [0.27, 5845]])

        else:
            task_kernel_params = gp_model.task_kernel_params
            # kernel_param = gp_model.kernel.params

        print("taks kernel:", task_kernel_params)
        X1_t = torch.from_numpy(X1).float()
        X2_t = torch.from_numpy(X2).float()
        Y1_t = torch.from_numpy(Y1).float()
        Y2_t = torch.from_numpy(Y2).float()

        gp_model = MultiTaskGaussianProcessImplicitSurfaces([X1_t, X2_t], [Y1_t, Y2_t], [T1,T2], kernel, task_kernel_params=task_kernel_params, sigma=sigma)
        gp_model.learning(max_iter=max_iter, lr=lr)

        mm2, ss2  = gp_model.predict(XX, T2)
        mean_zero = np.where(abs(mm2.T[0]) < 0.05)
        # mean_zero = np.where(abs(mm2.T[0]) < 0.1)
        surf_x    = xx.T[0][mean_zero]
        surf_y    = yy.T[0][mean_zero]
        var       = np.array(ss2.T[0][mean_zero])

        var_list.append(np.mean(var))
        simirality_list.append(gp_model.task_params_to_psd().numpy())


        if save_data:
            set_plt()
            plt.scatter(surf_x, surf_y, s=3, color=(1.0, 0.0, 0.0))
            plt.savefig('../data/high_low/mtgpis/estimate_surface/step{}.png'.format(i+1))
            plt.clf()

            # set_plt()
            # plt.plot(x2_0, y2_0, linewidth=3, color=(1.0, 0.0, 0.0))
            # plt.savefig('../data/high_low/mtgpis/true_surface/step{}.png'.format(i+1))
            # plt.clf()


        _, direrction = gp_model.predict_direction(torch.Tensor(current_po[:,None].T), T2)
        direrction = direrction.numpy()
        current_po   += alpha * direrction.T[0]

        normal, _ = gp_model.predict_direction(torch.Tensor(current_po[:,None].T), T2)
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


        # if i % 50 == 0 and i > 0:
        #     plt.plot(np.arange(len(var_list)), var_list)
        #     plt.show()
        ########################################## Plot ###################################################################
        if plot:
            ss2 = ss2.reshape(x.shape)
            z   = ss2.numpy()
            xyz = plt.pcolormesh(x, y, z, cmap='Greens', shading="gouraud", vmax=z.max(), vmin=z.min())

            plt.plot(x1_0, y1_0, linewidth=3, c="red",zorder=6)
            plt.plot(x2_0, y2_0, linewidth=3, c="lightskyblue",zorder=7)
            plt.plot(po_list[:, 0], po_list[:, 1], linewidth=5, c='orange', zorder=9)
            plt.scatter(po_list[-1, 0], po_list[-1, 1], c="darkorange", s=30, marker="o",zorder=10)
            plt.scatter(surf_x, surf_y, s=5, c="navy", zorder=8)
            # plt.scatter(X2[:,0], X2[:,1])
            set_plt()
            if save_movie:
                plt.savefig('../data/high_low/mtgpis/movie/step{}.png'.format(movie_num))
                movie_num += 1

            plt.draw()
            plt.pause(0.001)
            plt.clf()
        ########################################## Plot ###################################################################

    var_list = np.array(var_list)
    print("var_list:", var_list)
    print("simirality:", simirality_list)

    if save_data:
        np.save('../data/high_low/mtgpis/value/var', var_list)
        np.save('../data/high_low/mtgpis/value/simirality', simirality_list)


    plt.plot(np.arange(var_list.shape[0]), var_list)
    plt.show()



        # z = mm2.numpy()
        # xyz = plt.pcolormesh(x, y, z, cmap='Greens', shading="gouraud", vmax=z.max(), vmin=z.min())

        # plt.colorbar(xyz)
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(x, y, z, cmap='bwr', linewidth=0)
        # fig.colorbar(surf)
        # plt.show()