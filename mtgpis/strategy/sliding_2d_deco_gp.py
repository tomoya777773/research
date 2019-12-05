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
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# plt.style.use("ggplot")
from scipy.spatial import ConvexHull

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

def judge_square(po):
    if ( (0.15 < po[0] < 0.25) and (0.15 < po[1] < 0.45) ):
        return -1
    if ( (0.25 < po[0] < 0.35) and (0.15 < po[1] < 0.30) ):
        return -1
    if ( (0.35 < po[0] < 0.45) and (0.15 < po[1] < 0.45) ):
        return -1
    else:
        return 1

def set_plt():
    plt.xlim(-0.2, 0.8)
    plt.ylim(-0.2, 0.8)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.axes().set_aspect('equal', 'datalim')


if __name__=="__main__":
    # current_po = np.array([0.3, 0.4])
    current_po = np.array([0.4, 0.48])

    po_list    = np.array([current_po])

    # visual data
    N1 = 10
    x1_0 = (np.concatenate([np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1), np.ones(N1)*10]) ) * rate
    y1_0 = (np.concatenate([np.ones(N1)*10, np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1)]) ) * rate

    # true object
    N2 = 10
    x2_0 = np.concatenate([np.ones(N2)*15, np.linspace(15, 25, N2//2), np.ones(N2)*25, np.linspace(25, 35, N2), np.ones(N2)*35, np.linspace(35, 45, N2//2), np.ones(N2)*45, np.linspace(45, 15, N2)]) * rate
    y2_0 = np.concatenate([np.linspace(15, 45, N2), np.ones(N2//2)*45, np.linspace(45, 30, N2), np.ones(N2)*30, np.linspace(30, 45, N2), np.ones(N2//2)*45, np.linspace(45, 15, N2), np.ones(N2)*15]) * rate

    # test data
    x   = np.linspace(-20, 80, 400)[:, None] * rate
    y   = np.linspace(-20, 80, 400)[:, None] * rate
    # x   = np.linspace(0, 60, 200)[:, None] * rate
    # y   = np.linspace(0, 60, 200)[:, None] * rate
    x,y = np.meshgrid(x, y)
    xx  = x.ravel()[:, None]
    yy  = y.ravel()[:, None]
    XX  = np.concatenate([xx, yy], 1)
    XX  = torch.from_numpy(XX).float()

    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(111)

    X2 = np.array([current_po])
    Y2 = np.array([[1]])

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
                plt.savefig('../data/deco/gpis/movie/step{}.png'.format(movie_num))
                movie_num += 1

            plt.draw()
            plt.pause(0.001)
            plt.clf()
        ########################################## Plot ###################################################################

    # tactile data
    # X2 = np.array([current_po])
    # Y2 = np.array([[0]])
    X2 = np.append(X2, [current_po], axis=0)
    Y2 = np.append(Y2, [0])[:,None]
    X2 = np.append(X2, [current_po + np.array([-0.01, 0.02])], axis=0)
    Y2 = np.append(Y2, [1])[:,None]

    kernel = InverseMultiquadricKernelPytouch([kernel_param])

    var_list = []
    for i in range(145):
        print("=========================================================================")
        print("STEP: {}".format(i))

        X2_t = torch.from_numpy(X2).float()
        Y2_t = torch.from_numpy(Y2).float()

        gp_model = GaussianProcessImplicitSurfaces(X2_t, Y2_t, kernel, sigma=sigma)
        # gp_model.learning(max_iter=max_iter, lr=lr)
        # kernel_param = gp_model.kernel.params
        # sigma = gp_model.sigma

        mm2, ss2  = gp_model.predict(XX)
        # mean_zero = np.where(abs(mm2.T[0]) < 0.1)
        mean_zero = np.where(abs(mm2.T[0]) < 0.1)
        surf_x    = xx.T[0][mean_zero]
        surf_y    = yy.T[0][mean_zero]
        var       = np.array(ss2.T[0][mean_zero])

        var_list.append(np.mean(var))

        if save_data:
            set_plt()
            plt.scatter(surf_x, surf_y, s=3, color=(1.0, 0.0, 0.0))
            plt.savefig('../data/deco/gpis/estimate_surface/step{}.png'.format(i+1))
            plt.clf()

            # set_plt()
            # plt.plot(x2_0, y2_0, linewidth=3, color=(1.0, 0.0, 0.0))
            # plt.savefig('../data/deco/gpis/true_surface/step{}.png'.format(i+1))
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
            n = 1
            while n < 50:
                if judge_square(current_po) == 1:
                    X2 = np.append(X2, [current_po], axis=0)
                    Y2 = np.append(Y2, [0])[:,None]
                    po_list = np.append(po_list, [current_po], axis = 0)
                    break
                current_po += 0.0001 * normal.T[0]
                n += 1

            if judge_square(current_po) == -1:
                while True:
                    if judge_square(current_po) == 1:
                        X2 = np.append(X2, [current_po], axis=0)
                        Y2 = np.append(Y2, [0])[:,None]
                        po_list = np.append(po_list, [current_po], axis = 0)
                        break
                    current_po -= 0.0001 * direrction.T[0]


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
                plt.savefig('../data/deco/gpis/movie/step{}.png'.format(movie_num))
                movie_num += 1

            plt.draw()
            plt.pause(0.001)
            plt.clf()
        ########################################## Plot ###################################################################

    var_list = np.array(var_list)
    print("var_list:", var_list)

    if save_data:
        np.save('../data/deco/gpis/value/var', var_list)


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