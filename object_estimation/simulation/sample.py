#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

from kernel import InverseMultiquadricKernel, InverseMultiquadricKernelPytouch
from gpis import MultiTaskGaussianProcessImplicitSurfaces
from lib.judge_where import judge_ellipse

import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import mean_squared_error
import time
import copy

# hyper parameter
alpha        = 0.03
kernel_param = 0.4 #0.4, 0.6, 0.8
sigma        = 0.1 #0.1
z_limit      = 0.03
c            = 100
task         = 25
plot         = True
plot         = False

# save data
error_list, var_ave_list = [], []


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
    var0   = np.array(var0).reshape((N, N))
    z_t   = np.array(z_t).reshape((N, N))

    error   = np.sqrt(mean_squared_error(mean0_z, z_t))

    return [mean0_x, mean0_y, mean0_z], var0, error

def plot_estimated_surface(position, var):
    N = var
    # print N.min()
    # print N.max()
    ax.plot_surface(position[0], position[1], position[2],facecolors=cm.rainbow(N),
    linewidth=0, rstride=1, cstride=1, antialiased=False, shade=False, vmin=N.min(), vmax=N.max())

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
    # ax.set_xlim(-1.2, 1.2)
    # ax.set_ylim(-1.2, 1.2)
    # ax.set_zlim(-0.1, 1.1)
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
    r     = np.linspace(0.1, 0.6, N)

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
    outside_surface = np.load("../data/ellipse/ellipse_po_{}_2d.npy".format(task))
    radius = 0.2

    # Task 1
    X1 = np.load("../data/ellipse/ellipse_po_{}.npy".format(task))
    Y1 = np.zeros(len(X1))[:, None]
    T1 = 0
    # print X1.shape
    # tmp = copy.copy(X1)
    # tmp[:,2] += 0.1
    # # print X1
    # # print tmp
    # X1 = np.append(X1, tmp, axis=0)
    # Y1 = np.append(Y1, np.ones(len(Y1))[:, None], axis=0)


    # Task 2
    X2 = np.array([[-0.02, 0.02, 0.25],[-0.02, -0.02, 0.25], [0.02, -0.02, 0.25], [0.02, 0.02, 0.25]])
    # X2 = np.array([[-0.02, 0.02, 0.3],[-0.02, -0.02, 0.3], [0.02, -0.02, 0.3], [0.02, 0.02, 0.3]])
    # X2 = np.array([[-0.02, 0.02, 0.3],[-0.02, -0.02, 0.3], [0.02, -0.02, 0.3], [0.02, 0.02, 0.3]])
    # X2 = np.array([[-0.02, 0.02, 0.35],[-0.02, -0.02, 0.35], [0.02, -0.02, 0.35], [0.02, 0.02, 0.35]])
    # X2 = np.array([[-0.02, 0.02, 1.05],[-0.02, -0.02, 1.05], [0.02, -0.02, 1.05], [0.02, 0.02, 1.05]])

    Y2 = np.array([[1], [1], [1], [1]])
    T2 = 1
    # X1 = np.append(X1, X2, axis=0)
    # Y1 = np.append(Y1, np.ones(len(X2))[:, None], axis=0)

    # test data
    X, Y, Z, XX = make_test_data()
    XX = torch.from_numpy(XX).float()

    # choose kernel
    kernel = InverseMultiquadricKernelPytouch([kernel_param])

    # Show environment
    fig = plt.figure(figsize=(40, 40), dpi=50)

    # Go straight toward the object
    current_po    = np.array([-0.03, 0, 0.25])
    position_list = np.array([current_po])


    while True:
        if judge_ellipse([current_po[0], current_po[1], current_po[2]-0.002], radius) == -1:
            break

        current_po[2] -= 0.002
        position_list  = np.append(position_list, [current_po], axis = 0)

        ########################################## Plot ###################################################################
        if plot is True:
            ax = fig.add_subplot(111, projection='3d')
            # ax.scatter3D(X1[:,0], X1[:,1], X1[:,2])
            plot_environment(inside_surface, outside_surface)
            plot_path(position_list)
            # plt.show()
            plt.draw()
            plt.pause(0.001)
            plt.clf()
        ########################################## Plot ###################################################################

    current_po = current_po[:,None].T
    X2         = np.append(X2, current_po, axis=0)
    Y2         = np.append(Y2, [0])[:,None]


    for i in range(200):
            print "========================================================================="
            print "STEP: {}".format(i)

            if i == 0:
                task_kernel_params = torch.Tensor([[1, 10e-4], [10e-4, 1]])
                max_iter = 400
            else:
                task_kernel_params = gp_model.task_kernel_params
                max_iter = 50

            # task_kernel_params = torch.Tensor([[1, 10e-4], [10e-4, 1]])
            # max_iter = 400

            print "taks kernel:", task_kernel_params
            X1_t = torch.from_numpy(X1).float()
            X2_t = torch.from_numpy(X2).float()
            Y1_t = torch.from_numpy(Y1).float()
            Y2_t = torch.from_numpy(Y2).float()

            gp_model = MultiTaskGaussianProcessImplicitSurfaces([X1_t, X2_t], [Y1_t, Y2_t], [T1,T2],
                                                    kernel, task_kernel_params, c=c, sigma=sigma, z_limit=z_limit)

            gp_model.learning(max_iter=max_iter)

            normal, direrction = gp_model.predict_direction(torch.Tensor(current_po), T2)
            normal, direrction = normal.numpy(), direrction.numpy()

            current_po        += alpha * direrction.T

            mean, var = gp_model.predict(XX, T2)

            estimated_surface, var, error = get_object_position(X, Y, Z, mean, var, radius)
            error_list.append(error)
            var_ave_list.append(np.mean(var))
            print "error:", error
            print "var:", np.mean(var)
            np.save("../data/mtgpis/ellipse{0}/step_{1}".format(task, i), [np.array(mean), np.array(var), np.array(gp_model.task_kernel)])

            ########################################## Plot ###################################################################
            if plot is True:
                ax = fig.add_subplot(111, projection='3d')
                plot_environment(inside_surface, outside_surface)
                plot_estimated_surface(estimated_surface, var)
                plot_path(position_list)
                plt.draw()
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
            position_list      = np.append(position_list, [current_po[0]], axis = 0)

    mean, var = gp_model.predict(XX, T2)
    estimated_surface, var, error = get_object_position(X, Y, Z, mean, var, radius)
    error_list.append(error)
    var_ave_list.append(np.mean(var))
    position_list = np.append(position_list, [current_po[0]], axis = 0)

    print "error:", error_list
    print "var:", var_ave_list
    np.save("../data/mtgpis/ellipse{}/position".format(task), position_list)
    np.save("../data/mtgpis/ellipse{}/error".format(task), error_list)
    np.save("../data/mtgpis/ellipse{}/var".format(task), var_ave_list)


    ########################################## Plot ###################################################################
    ax = fig.add_subplot(111, projection='3d')
    plot_environment(inside_surface, outside_surface)
    plot_estimated_surface(estimated_surface, var)
    plot_path(position_list)

    plt.show()
    ########################################## Plot ###################################################################


            # if i % 50 == 0:
            #     mean, var = gp_model.predict(XX, T2)
            #     mean_x, mean_y, mean_z, var, error, var_ave = get_object_position(X, Y, Z, mean, var, radius)
            #     np.save("../data/mtgpis_200_per5/mtgpis_task1_{}".format(i), [mean_x, mean_y, mean_z, var, error, var_ave])
