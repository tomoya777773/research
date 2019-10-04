#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

from kernel import InverseMultiquadricKernel
from gpis import GaussianProcessImplicitSurfaces
from lib.judge_where import judge_ellipse

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import mean_squared_error
# import time


# hyper parameter
alpha = 0.03
kernel_param = 0.4


def get_object_position(x,y,z, mean, var, r):
    mean  = mean.reshape(x.shape)
    var   = var.reshape(x.shape)
    mean0 = np.argmin(np.abs(mean), axis = 2)

    mean0_x, mean0_y, mean0_z, var0, z_t = [], [], [], [], []
    for i in range(len(x)):
        for j in range(len(x)):
            mean0_x.append(x[i][j][mean0[i][j]])
            mean0_y.append(y[i][j][mean0[i][j]])
            mean0_z.append(z[i][j][mean0[i][j]])
            var0.append(var[i][j][mean0[i][j]])
            tmp = np.sqrt(np.linalg.norm(r**2 - x[i][j][mean0[i][j]]**2 / 4 - y[i][j][mean0[i][j]]**2 / 4))
            z_t.append(tmp)

    N = len(x)
    mean0_x = np.array(mean0_x).reshape((N, N))
    mean0_y = np.array(mean0_y).reshape((N, N))
    mean0_z = np.array(mean0_z).reshape((N, N))
    var0    = np.array(var0).reshape((N, N))
    z_t     = np.array(z_t).reshape((N, N))

    error   = np.sqrt(mean_squared_error(mean0_z, z_t))
    var_ave = np.mean(var0)

    return [mean0_x, mean0_y, mean0_z], var0, error, var_ave

def plot_estimated_surface(position, var):
    N = var

    ax.plot_surface(position[0], position[1], position[2],facecolors=cm.rainbow(N),
    linewidth=0, rstride=1, cstride=1, antialiased=False, shade=False)

    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array(N)
    # plt.colorbar(m)

def plot_path(position):
    start_po = position[0]
    current_po = position[-1]

    ax.plot(position[:,0], position[:,1], position[:,2],  "o-", color="black", ms=20, mew=5, linewidth=10)
    ax.plot([current_po[0]], [current_po[1]], [current_po[2]],  "o-", color="#00aa00", ms=30, mew=5)
    ax.plot([start_po[0]], [start_po[1]], [start_po[2]],  "o-", color="red", ms=30, mew=5)

def plot_environment(in_surf, out_surf):
    plt.gca().patch.set_facecolor('white')
    ax._axis3don = False

    ax.view_init(30, 60)
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.1, 0.3)

    ax.plot_surface(in_surf[0], in_surf[1], in_surf[2], color="green",
    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.1)
    ax.plot_surface(out_surf[0], out_surf[1], out_surf[2], color="blue",
    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.1)

def make_test_data():
    N = 40
    theta = np.linspace(-np.pi, np.pi, N)
    phi   = np.linspace(0, np.pi/2, N)
    r     = np.linspace(0.01, 0.41, N)

    THETA, PHI, R = np.meshgrid(theta, phi, r)

    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)

    x_test = np.ravel(X)[:,None]
    y_test = np.ravel(Y)[:,None]
    z_test = np.ravel(Z)[:,None]

    return X, Y, Z, np.concatenate([x_test, y_test, z_test], 1)


if __name__=="__main__":

    # object position
    inside_surface  = np.load("../data/ellipse/ellipse_po_20_2d.npy")
    outside_surface = np.load("../data/ellipse/ellipse_po_25_2d.npy")
    radius = 0.2

    # Task 1
    X1 = np.load("../data/ellipse/ellipse_po_25.npy")
    Y1 = np.zeros(len(X1))[:, None]
    T1 = 0

    # Task 2
    X2 = np.array([[-0.04, 0.04, 0.3],[0.03, -0.01, 0.3]])
    Y2 = np.array([[1], [1]])
    T2 = 1

    # test data
    X, Y, Z, XX = make_test_data()

    # GPIS model
    kernel   = InverseMultiquadricKernel([kernel_param])

    # Show environment
    fig = plt.figure(figsize=(40, 40), dpi=50)

    # Go straight toward the object
    current_po    = np.array([0.01, 0.01, 0.25])
    position_list = np.array([current_po])


    while True:
        if judge_ellipse([current_po[0], current_po[1], current_po[2]-0.002], radius) == -1:
            break

        current_po[2] -= 0.002
        position_list  = np.append(position_list, [current_po], axis = 0)

        ########################################## Plot ###################################################################
        ax = fig.add_subplot(111, projection='3d')

        plot_environment(inside_surface, outside_surface)
        plot_path(position_list)
        plt.pause(0.001)
        plt.clf()
        ########################################## Plot ###################################################################


    current_po = current_po[:,None].T
    X2         = np.append(X2, current_po, axis=0)
    Y2         = np.append(Y2, [0])[:,None]


    error_list, var_ave_list = [], []
    for i in range(201):
            print "========================================================================="
            print "STEP: {}".format(i)

            gp_model           = GaussianProcessImplicitSurfaces(X2, Y2, kernel, c=10)
            normal, direrction = gp_model.predict_direction(current_po)
            current_po        += alpha * direrction.T
            position_list      = np.append(position_list, [current_po[0]], axis = 0)

            mean, var = gp_model.predict(XX)

            estimated_surface, var, error, var_ave = get_object_position(X, Y, Z, mean, var, radius)

            error_list.append(error)
            var_ave_list.append(var_ave)

            ########################################## Plot ###################################################################
            ax = fig.add_subplot(111, projection='3d')

            plot_environment(inside_surface, outside_surface)
            plot_estimated_surface(estimated_surface, var)
            plot_path(position_list)

            plt.pause(0.001)
            plt.clf()
            ########################################## Plot ###################################################################

            judge = judge_ellipse(current_po[0], radius)

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


    mean, var = gp_model.predict(XX)
    estimated_surface, var, error, var_ave = get_object_position(X, Y, Z, mean, var, radius)

    error_list.append(error)
    var_ave_list.append(var_ave)
    position_list = np.append(position_list, [current_po[0]], axis = 0)

    # np.save("../data/gpis_200_per5/gpis_position", position_list)
    # np.save("../data/gpis_error_200", error_list)
    # np.save("../data/gpis_var_ave_200", var_ave_list)

    ########################################## Plot ###################################################################
    ax = fig.add_subplot(111, projection='3d')
    plot_environment(inside_surface, outside_surface)
    plot_estimated_surface(estimated_surface, var)
    plot_path(position_list)

    plt.show()
    ########################################## Plot ###################################################################



    # if i % 50 == 0 and i > 0:
    #     mean, var = gp_model.predict(XX)
    #     mean_x, mean_y, mean_z, var, error, var_ave = get_object_position(X, Y, Z, mean, var, radius)
    #     np.save("../data/gpis_200_per5/gpis_{}".format(i), [mean_x, mean_y, mean_z, var, error, var_ave])

