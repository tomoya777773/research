#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
from kernel import InverseMultiquadricKernel

from gpis import GaussianProcessImplicitSurfaces
from mtgpis import MultiTaskGaussianProcessImplicitSurfaces

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
plt.style.use("ggplot")

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import time

def mean_0_position(x,y,z, mm, ss):
    mm = mm.reshape(x.shape)
    ss = ss.reshape(x.shape)
    mm0 = np.argmin(np.abs(mm), axis = 2)
    # print mm0
    # print mm0.shape
    mm0_x, mm0_y, mm0_z, ss0 = [], [], [], []
    for i in range(len(x)):
        for j in range(len(x)):

            mm0_x.append(x[i][j][0])
            mm0_y.append(y[i][j][0])
            mm0_z.append(z[i][j][mm0[i][j]])
            ss0.append(ss[i][j][mm0[i][j]])
    mm0_x = np.array(mm0_x)
    mm0_y = np.array(mm0_y)
    mm0_z = np.array(mm0_z)
    ss0   = np.array(ss0)

    N = len(x)
    mm0_x = mm0_x.reshape((N, N))
    mm0_y = mm0_y.reshape((N, N))
    mm0_z = mm0_z.reshape((N, N))
    ss0   = ss0.reshape((N, N))
    # print ss0

    return mm0_x, mm0_y, mm0_z, ss0

def mtgpis_plot_3d(X1, X2, mm0_po, ss, x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.scatter3D(mm2_x, mm2_y, mm2_z)

    dx, dy = 0.05, 0.05
    # zz = ss[:-1, :-1]
    levels = MaxNLocator(nbins=30).tick_values(z.min(), z.max())
    cmap   = plt.get_cmap('Reds')

    ax.scatter3D(X1[:, 0], X1[:, 1], X1[:, 2], alpha=0.1)
    ax.scatter3D(X2[:, 0], X2[:, 1], X2[:, 2], s=100, c="green", edgecolors="red")

    print ss.max()
    print ss.min()
    N = ss/ss.max()
    surf = ax.plot_surface(mm0_po[0], mm0_po[1], mm0_po[2], facecolors=cm.jet(N), linewidth=0, rstride=1, cstride=1, antialiased=True)

    # cf = ax.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1] + dy/2., zz, levels=levels, cmap=cmap)

    m = cm.ScalarMappable(cmap=cm.PiYG_r)
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(ss)
    cbar = plt.colorbar(m)

    # ax.set_xlim(0,1)
    ax.set_zlim(-0.3, 0.3)

    # fig.colorbar(cf)
    plt.show()


if __name__=="__main__":

    # # 入力ノイズと出力ノイズの初期設定
    # data_sigma_x = 0.001
    # data_sigma_y = 0.001

    X1 = np.load("../data/sphere_po_30.npy")
    Y1 = np.zeros(len(X1))[:, None]
    T1 = 0

    X2 = np.load("../data/sphere_po_28.npy")
    Y2 = np.zeros(len(X2))[:, None]
    T2 = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X1[:, 0], X1[:, 1], X1[:, 2], alpha=0.1)
    ax.scatter3D(X2[:, 0], X2[:, 1], X2[:, 2])
    plt.show()

    N = 20
    x   = np.linspace(-0.3, 0.3, N, dtype=np.float32)
    y   = np.linspace(-0.3, 0.3, N, dtype=np.float32)
    z   = np.linspace(0, 0.3, N, dtype=np.float32)
    x,y,z = np.meshgrid(x, y, z)
    xx  = x.ravel()[:, None]
    yy  = y.ravel()[:, None]
    zz  = z.ravel()[:, None]

    XX  = np.concatenate([xx, yy, zz], 1)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(xx, yy, zz, alpha=0.1)
    # plt.show()

    # カーネルの選択とパラメータの初期化
    kernel = InverseMultiquadricKernel([0.3])

    # マルチガウス過程回帰のモデルを生成


    # GPIS
    model1 = GaussianProcessImplicitSurfaces(X2, Y2, kernel)
    mm2, ss2 = model1.predict(XX)
    mm2_x, mm2_y, mm2_z, ss2 = mean_0_position(x,y,z, mm2, ss2)
    mtgpis_plot_3d(X1, X2, [mm2_x, mm2_y, mm2_z], ss2, x,y,z)


    # MTGPIS
    model  = MultiTaskGaussianProcessImplicitSurfaces([X1,X2], [Y1,Y2], [T1,T2], kernel)

    # MTGP before optimize paramater
    print("========== STEP 1 ==========")

    mm1, ss1 = model.predict(XX, T1)
    mm2, ss2 = model.predict(XX, T2)
    mm2_x, mm2_y, mm2_z, ss2 = mean_0_position(x,y,z, mm2, ss2)
    mtgpis_plot_3d(X1, X2, [mm2_x, mm2_y, mm2_z], ss2, x,y,z)

    # MTGP after optimize paramater
    print("========== STEP 2 ==========")
    print("------------------------------")
    # print("aaaaaaaaaaaaaaaaa",model.task_to_psd_matrix())
    model.learn_params()
    # print("bbbbbbbbbbbbbbbbb",model.task_to_psd_matrix())
    print("-----------------------------")

    # mm1, ss1 = model.predict(XX, T1)
    mm2, ss2 = model.predict(XX, T2)
    mm2_x, mm2_y, mm2_z, ss2 = mean_0_position(x,y,z, mm2, ss2)

    mtgpis_plot_3d(X1, X2, [mm2_x, mm2_y, mm2_z], ss2, x,y,z)