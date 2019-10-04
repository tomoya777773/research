#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

from kernel import InverseMultiquadricKernel
from gpis import MultiTaskGaussianProcessImplicitSurfaces

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("ggplot")

# import time
from sklearn.metrics import mean_squared_error

def mean_0_position(x,y,z, mm, ss, r):
    # r = 0.23
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
    print error
    print var_ave

    return mm0_x, mm0_y, mm0_z, ss0, error, var_ave

def mtgpis_plot_3d(mm0_po, ss):
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.scatter3D(mm2_x, mm2_y, mm2_z)

    # ax.scatter3D(X1[:, 0], X1[:, 1], X1[:, 2], alpha=0.1)
    # ax.scatter3D(X2[:, 0], X2[:, 1], X2[:, 2], s=100, c="green", edgecolors="red")


    # N = (ss - ss.min()) / (ss.max() - ss.min())
    N = ss
    ax.plot_surface(mm0_po[0], mm0_po[1], mm0_po[2], facecolors=cm.rainbow(N), linewidth=0, rstride=1, cstride=1, antialiased=False)
    # ax.plot_surface(x,y,z, facecolors=cm.jet(N), linewidth=0, rstride=1, cstride=1, antialiased=True)

    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array(N)
    plt.colorbar(m)

    # ax.set_xlim(-0.6, 0.6)
    # ax.set_ylim(-0.6, 0.6)
    # ax.set_zlim(-0.6, 0.6)

    # fig.colorbar(cf)
    # plt.show()

def plot_sphere(r):
    # theta, phi = np.linspace(0, np.pi, 20), np.linspace(0, np.pi/2, 20)
    # theta, phi = np.linspace(np.pi/8, np.pi/8*7, 1), np.linspace(np.pi/8, np.pi/9*4, 1)
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

def plot_environment(position_list, X):
    current_po = position_list[-1]

    ax.view_init(60, 60)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)

    ax.plot(position_list[:,0], position_list[:,1], position_list[:,2],  "o-", color="black", ms=20, mew=5)
    ax.plot([current_po[0]], [current_po[1]], [current_po[2]],  "o-", color="#00aa00", ms=30, mew=5)
    ax.scatter3D(X1[:, 0], X1[:, 1], X1[:, 2], s=100, alpha=0.1)

    # plt.pause(0.001)
    # plt.clf()

def judge_ellipse(position, r):
    x,y,z = position
    judge = (x**2 / 4 + y**2 / 4 + z**2) - r**2
    if judge < 0: # in
        return -1
    elif judge > 0 and round(judge, 4) == 0: # on
        return 0
    else: # out
        return 1

if __name__=="__main__":
    # # 入力ノイズと出力ノイズの初期設定
    # data_sigma_x = 0.001
    # data_sigma_y = 0.001

    # Task 1
    # X1 = np.load("../data/ellipse_po_22.npy")

    X1 = np.load("../data/ellipse_po_21_25.npy")
    Y1 = np.zeros(len(X1))[:, None]
    T1 = 0


    # Task 2
    X2 = np.array([[0.02, 0.02, 0.3],[-0.02, -0.02, 0.3]])
    Y2 = np.array([[1], [1]])
    T2 = 1
    radius = 0.2



    fig = plt.figure(figsize=(40, 40), dpi=50)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(X1[:, 0], X1[:, 1], X1[:, 2], s=100, alpha=0.1)
    plot_sphere(radius)
    plt.show()

    # test data
    N = 20
    theta = np.linspace(-np.pi, np.pi, N)
    phi   = np.linspace(0, np.pi/2, N)
    # r     = np.linspace(0.15, 0.45, N)
    r     = np.linspace(0.12, 0.42, N)


    THETA, PHI, R = np.meshgrid(theta, phi, r)
    # R = np.cos(PHI) * R

    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)

    x_test = np.ravel(X)[:,None]
    y_test = np.ravel(Y)[:,None]
    z_test = np.ravel(Z)[:,None]

    XX = np.concatenate([x_test, y_test, z_test], 1)


    kernel = InverseMultiquadricKernel([0.4])


    # Show environment
    fig = plt.figure(figsize=(40, 40), dpi=50)

    # Go straight toward the object
    current_po    = np.array([0.01, 0.01, 0.22])
    position_list = np.array([current_po])

    while True:
        if judge_ellipse([current_po[0], current_po[1], current_po[2]-0.002], radius) == -1:
            break

        current_po[2] -= 0.002
        position_list  = np.append(position_list, [current_po], axis = 0)

        ax = fig.add_subplot(111, projection='3d')
        plot_sphere(radius)
        plot_environment(position_list, X1)
        plt.pause(0.001)
        plt.clf()
        # plt.show()

    current_po = current_po[:,None].T
    X2         = np.append(X2, current_po, axis=0)
    Y2         = np.append(Y2, [0])[:,None]


    alpha = 0.03
    error_list, var_ave_list = [], []

    for i in range(30):
            print "========================================================================="
            print "STEP: {}".format(i)
            # print Y2

            if i == 0:
                tmp = np.array([[1, 0.01], [0.01, 1]])
                task_kernel = np.triu(tmp.T).T
            else:
                task_kernel = gp_model.task_kernel_params

            gp_model = MultiTaskGaussianProcessImplicitSurfaces([X1,X2], [Y1,Y2], [T1,T2], kernel, task_kernel)
            gp_model.learn_params()
            # gp_model.learn_params_plus_kernel_params()

            normal, direrction = gp_model.predict_direction(current_po, T2)
            current_po        += alpha * direrction.T
            position_list      = np.append(position_list, [current_po[0]], axis = 0)

            print "normal:", normal
            print "direction:", direrction

            mm2, ss2 = gp_model.predict(XX, T2)
            mm2_x, mm2_y, mm2_z, ss2, error, var_ave = mean_0_position(X, Y, Z, mm2, ss2, radius)
            error_list.append(error)
            var_ave_list.append(var_ave)

            ########################################## Plot ###################################################################
            ax = fig.add_subplot(111, projection='3d')
            plot_environment(position_list, X1)
            mtgpis_plot_3d([mm2_x, mm2_y, mm2_z], ss2)

            plt.pause(0.001)
            plt.clf()
            ########################################## Plot ###################################################################


            judge = judge_ellipse(current_po[0], radius)
            print "judge:", judge

            if judge == 0:
                X2 = np.append(X2, current_po, axis=0)
                Y2 = np.append(Y2, [0])[:,None]
                continue

            elif judge == 1:
                # X2 = np.append(X2, current_po, axis=0)
                # Y2 = np.append(Y2, [1])[:,None]
                while True:
                    if judge_ellipse((current_po - 0.0005 * normal.T)[0], radius) == -1:
                        X2 = np.append(X2, current_po, axis=0)
                        Y2 = np.append(Y2, [0])[:,None]
                        break
                    current_po -= 0.0005 * normal.T

            elif judge == -1:
                while True:
                    if judge_ellipse(current_po[0], radius) == 1:
                        X2 = np.append(X2, current_po, axis=0)
                        Y2 = np.append(Y2, [0])[:,None]
                        break
                    current_po += 0.0005 * normal.T

    mm2, ss2 = gp_model.predict(XX, T2)
    mm2_x, mm2_y, mm2_z, ss2, error, var_ave = mean_0_position(X, Y, Z, mm2, ss2, radius)
    error_list.append(error)
    var_ave_list.append(var_ave)
    print "error:", error_list
    print "var:", var_ave_list
    np.save("../data/mtgpis_error_100_cos", error_list)
    np.save("../data/mtgpis_var_ave_100_cos", var_ave_list)

    ########################################## Plot ###################################################################
    ax = fig.add_subplot(111, projection='3d')
    plot_environment(position_list, X1)
    mtgpis_plot_3d([mm2_x, mm2_y, mm2_z], ss2)

    plt.show()
    ########################################## Plot ###################################################################


