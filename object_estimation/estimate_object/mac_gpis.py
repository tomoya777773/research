#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

from kernel import InverseMultiquadricKernel
from gpis import GaussianProcessImplicitSurfaces

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("ggplot")

from sklearn.metrics import mean_squared_error


def mean_0_position(x,y,z, mm, ss, r):
    mm = mm.reshape(x.shape)
    ss = ss.reshape(x.shape)
    mm0 = np.argmin(np.abs(mm), axis = 2)

    mm0_x, mm0_y, mm0_z, ss0, z_t = [], [], [], [], []
    for i in range(len(x)):
        for j in range(len(x)):
            mm0_x.append(x[i][j][mm0[i][j]])
            mm0_y.append(y[i][j][mm0[i][j]])
            mm0_z.append(z[i][j][mm0[i][j]])
            ss0.append(ss[i][j][mm0[i][j]])
            tmp = np.sqrt(np.linalg.norm(r**2 - x[i][j][mm0[i][j]]**2 / 4 - y[i][j][mm0[i][j]]**2 / 4))
            z_t.append(tmp)

    N = len(x)
    mm0_x = np.array(mm0_x).reshape((N, N))
    mm0_y = np.array(mm0_y).reshape((N, N))
    mm0_z = np.array(mm0_z).reshape((N, N))
    ss0   = np.array(ss0).reshape((N, N))
    z_t   = np.array(z_t).reshape((N, N))

    error   = np.sqrt(mean_squared_error(mm0_z, z_t))
    var_ave = np.mean(ss0)
    # print error
    # print var_ave

    return mm0_x, mm0_y, mm0_z, ss0, error, var_ave

def mtgpis_plot_3d(mm0_po, ss):
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.scatter3D(mm2_x, mm2_y, mm2_z)

    # ax.scatter3D(X1[:, 0], X1[:, 1], X1[:, 2], alpha=0.1)
    # ax.scatter3D(X2[:, 0], X2[:, 1], X2[:, 2], s=100, c="green", edgecolors="red")


    N = ss
    ax.plot_surface(mm0_po[0], mm0_po[1], mm0_po[2], facecolors=cm.rainbow(N), linewidth=0, rstride=1, cstride=1, antialiased=False)
    # ax.plot_surface(x,y,z, facecolors=cm.jet(N), linewidth=0, rstride=1, cstride=1, antialiased=True)

    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array(N)
    # plt.colorbar(m)

    # ax.set_xlim(0,1)
    # ax.set_xlim(-0.3, 0.3)
    # ax.set_ylim(-0.3, 0.3)
    # ax.set_zlim(-0.3, 0.3)

    # fig.colorbar(cf)
    # plt.show()

def plot_sphere(r):
    theta, phi = np.linspace(-np.pi, np.pi, 20), np.linspace(0, np.pi/4, 20)

    THETA, PHI = np.meshgrid(theta, phi)
    R = np.cos(PHI) * r*2
    X = R * np.sin(PHI) * np.cos(THETA) * 2
    Y = R * np.sin(PHI) * np.sin(THETA) * 2
    Z = R * np.cos(PHI) - r


    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)

    # plt.show()

def judge_ellipse(position, r):
    x,y,z = position
    judge = (x**2 / 4 + y**2 / 4 + z**2) - r**2
    # print(judge)
    if judge < 0: # in
        return -1
    elif judge > 0 and round(judge, 4) == 0: # on
        return 0
    else: # out
        return 1

def plot_environment(position_list, X):
    current_po = position_list[-1]

    ax.view_init(40, 40)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0, 0.6)

    ax.plot(position_list[:,0], position_list[:,1], position_list[:,2],  "o-", color="black", ms=20, mew=5)
    ax.plot([current_po[0]], [current_po[1]], [current_po[2]],  "o-", color="#00aa00", ms=30, mew=5)
    ax.scatter3D(X1[:, 0], X1[:, 1], X1[:, 2], s=100, alpha=0.1)


if __name__=="__main__":
    # # 入力ノイズと出力ノイズの初期設定
    # data_sigma_x = 0.001
    # data_sigma_y = 0.001

    # Task 1
    X1 = np.load("../data/ellipse/ellipse_po_21.npy")
    Y1 = np.zeros(len(X1))[:, None]
    T1 = 0

    # Task 2
    X2 = np.array([[-0.04, 0.04, 0.5],[0.03, -0.08, 0.5]])
    Y2 = np.array([[1], [1]])
    T2 = 1
    radius = 0.2

    # test data
    N = 15
    theta = np.linspace(-np.pi, np.pi, N)
    phi   = np.linspace(0, np.pi/2, N)
    # r     = np.linspace(0.15, 0.45, N)
    r     = np.linspace(0.01, 0.41, N)

    THETA, PHI, R = np.meshgrid(theta, phi, r)
    # R = np.cos(PHI) * R

    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)

    x_test = np.ravel(X)[:,None]
    y_test = np.ravel(Y)[:,None]
    z_test = np.ravel(Z)[:,None]

    XX = np.concatenate([x_test, y_test, z_test], 1)


    kernel = InverseMultiquadricKernel([0.45])
    gp_model = GaussianProcessImplicitSurfaces(X2, Y2, kernel)


    # Show environment
    fig = plt.figure(figsize=(40, 40), dpi=50)
    # ax.scatter3D(X1[:, 0], X1[:, 1], X1[:, 2], s=100, alpha=0.1)

    # Go straight toward the object
    current_po    = np.array([0.01, 0.01, 0.22])
    position_list = np.array([current_po])

    while True:
        if judge_ellipse([current_po[0], current_po[1], current_po[2]-0.002], radius) == -1:
            break

        current_po[2] -= 0.002
        position_list  = np.append(position_list, [current_po], axis = 0)

        # ax = fig.add_subplot(111, projection='3d')
        # plot_sphere(radius)
        # plot_environment(position_list, X1)
        # plt.pause(0.001)
        # plt.clf()

    current_po = current_po[:,None].T
    X2         = np.append(X2, current_po, axis=0)
    Y2         = np.append(Y2, [0])[:,None]


    alpha = 0.03
    error_list, var_ave_list = [], []
    for i in range(201):
            # print "========================================================================="
            # print "STEP: {}".format(i)

            gp_model           = GaussianProcessImplicitSurfaces(X2, Y2, kernel)
            normal, direrction = gp_model.predict_direction(current_po)
            current_po        += alpha * direrction.T
            position_list      = np.append(position_list, [current_po[0]], axis = 0)

            mm2, ss2 = gp_model.predict(XX)
            mm2_x, mm2_y, mm2_z, ss2, error, var_ave = mean_0_position(X, Y, Z, mm2, ss2, radius)
            error_list.append(error)
            var_ave_list.append(var_ave)

            ########################################## Plot ###################################################################
            ax = fig.add_subplot(111, projection='3d')

            ax.patch.set_facecolor('white') 
            ax.patch.set_alpha(0)
            ax.w_xaxis.set_pane_color((0., 0., 0., 0.))
            ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
            ax.w_zaxis.set_pane_color((0., 0., 0., 0.))
            ax.tick_params(color='white')
            ax.tick_params(labelbottom='off',
                            labelleft='off',
                            labelright='off',
                            labeltop='off',
                            bottom='off',
                            left='off',
                            right='off',
                            top='off')
            # plt.tick_params(labelbottom='off')
            # ax.tick_params(direction = "inout", length = 0, colors = "blue")
            plt.tick_params(labelbottom='off', labelleft='off', labelright='off', labeltop='off')
            plot_environment(position_list, X1)
            mtgpis_plot_3d([mm2_x, mm2_y, mm2_z], ss2)


            plt.pause(0.001)
            plt.clf()
            ########################################## Plot ###################################################################


            # if i % 50 == 0 and i > 0:
            #     mm2, ss2 = gp_model.predict(XX)
            #     mm2_x, mm2_y, mm2_z, ss2, error, var_ave = mean_0_position(X, Y, Z, mm2, ss2, radius)
            #     np.save("../data/gpis_200_per5/gpis_{}".format(i), [mm2_x, mm2_y, mm2_z, ss2, error, var_ave])

            judge = judge_ellipse(current_po[0], radius)
            # print "judge:", judge

            if judge == 0:
                X2 = np.append(X2, current_po, axis=0)
                Y2 = np.append(Y2, [0])[:,None]
                continue

            elif judge == 1:
                # X2 = np.append(X2, current_po, axis=0)
                # Y2 = np.append(Y2, [1])[:,None]
                while True:
                    if judge_ellipse((current_po - 0.0001 * normal.T)[0], radius) == -1:
                        X2 = np.append(X2, current_po, axis=0)
                        Y2 = np.append(Y2, [0])[:,None]
                        break
                    current_po -= 0.0001 * normal.T

            elif judge == -1:
                while True:
                    if judge_ellipse(current_po[0], radius) == 1:
                        X2 = np.append(X2, current_po, axis=0)
                        Y2 = np.append(Y2, [0])[:,None]
                        break
                    current_po += 0.0001 * normal.T


    mm2, ss2 = gp_model.predict(XX)
    mm2_x, mm2_y, mm2_z, ss2, error, var_ave = mean_0_position(X, Y, Z, mm2, ss2, radius)
    error_list.append(error)
    var_ave_list.append(var_ave)
    position_list      = np.append(position_list, [current_po[0]], axis = 0)

    # print "error:", error_list
    # print "var:", var_ave_list
    # np.save("../data/gpis_200_per5/gpis_position", position_list)
    # print position_list
    # np.save("../data/gpis_error_200", error_list)
    # np.save("../data/gpis_var_ave_200", var_ave_list)


    ########################################## Plot ###################################################################
    fig = plt.figure(figsize=(40, 40), dpi=50)

    ax = fig.add_subplot(111, projection='3d')
    plot_environment(position_list, X1)
    mtgpis_plot_3d([mm2_x, mm2_y, mm2_z], ss2)

    plt.show()
    ########################################## Plot ###################################################################























    # N = 20
    # x   = np.linspace(-0.25, 0.25, N, dtype=np.float32)
    # y   = np.linspace(-0.25, 0.25, N, dtype=np.float32)
    # z   = np.linspace(0, 0.25, N, dtype=np.float32)
    # x,y,z = np.meshgrid(x, y, z)
    # xx  = x.ravel()[:, None]
    # yy  = y.ravel()[:, None]
    # zz  = z.ravel()[:, None]

    # XX  = np.concatenate([xx, yy, zz], 1)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(xx, yy, zz, alpha=0.1)
    # plt.show()

    # # カーネルの選択とパラメータの初期化
    # kernel = InverseMultiquadricKernel([0.3])

    # # マルチガウス過程回帰のモデルを生成


    # # GPIS
    # model1 = GPISEstimation(X2, Y2, kernel)
    # mm2, ss2 = model1.predict(XX)
    # mm2_x, mm2_y, mm2_z, ss2 = mean_0_position(x,y,z, mm2, ss2)
    # mtgpis_plot_3d(X1, X2, [mm2_x, mm2_y, mm2_z], ss2, x,y,z)


    # # MTGPIS
    # model  = MTGPISEstimation([X1,X2], [Y1,Y2], [T1,T2], kernel)

    # # MTGP before optimize paramater
    # print("========== STEP 1 ==========")

    # mm1, ss1 = model.predict(XX, T1)
    # mm2, ss2 = model.predict(XX, T2)
    # mm2_x, mm2_y, mm2_z, ss2 = mean_0_position(x,y,z, mm2, ss2)
    # mtgpis_plot_3d(X1, X2, [mm2_x, mm2_y, mm2_z], ss2, x,y,z)

    # # MTGP after optimize paramater
    # print("========== STEP 2 ==========")
    # print("------------------------------")
    # print("aaaaaaaaaaaaaaaaa",model.task_to_psd_matrix())
    # model.learn_params()
    # print("bbbbbbbbbbbbbbbbb",model.task_to_psd_matrix())
    # print("-----------------------------")

    # # mm1, ss1 = model.predict(XX, T1)
    # mm2, ss2 = model.predict(XX, T2)
    # mm2_x, mm2_y, mm2_z, ss2 = mean_0_position(x,y,z, mm2, ss2)

    # mtgpis_plot_3d(X1, X2, [mm2_x, mm2_y, mm2_z], ss2, x,y,z)