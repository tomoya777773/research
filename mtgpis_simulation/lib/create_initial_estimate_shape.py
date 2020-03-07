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
max_iter     = 1000
lr           = 0.0001
sigma        = torch.tensor(-5.168)


plot = True
# plot = False
save_data = True
save_data = False
save_movie = True
# save_movie = False



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
    # current_po = np.array([0.4, 0.55])
    current_po = np.array([0.3, 0.5]) # high high
    current_po = np.array([0.3, 0.55]) # low low
    current_po = np.array([0.3, 0.55]) # high low
    current_po = np.array([0.25, 0.55]) # low high
    current_po = np.array([0.4, 0.55]) # deco

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
    x2_0 = np.concatenate([np.ones(N2)*15, np.linspace(15, 25, N2//2), np.ones(N2)*25, np.linspace(25, 35, N2), np.ones(N2)*35, np.linspace(35, 45, N2//2), np.ones(N2)*45, np.linspace(45, 15, N2)]) * rate
    y2_0 = np.concatenate([np.linspace(15, 45, N2), np.ones(N2//2)*45, np.linspace(45, 30, N2), np.ones(N2)*30, np.linspace(30, 45, N2), np.ones(N2//2)*45, np.linspace(45, 15, N2), np.ones(N2)*15]) * rate

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
    ax = fig.add_subplot(111)


    kernel = InverseMultiquadricKernelPytouch([kernel_param])


    X1_t = torch.from_numpy(X1).float()
    Y1_t = torch.from_numpy(Y1).float()

    gp_model = GaussianProcessImplicitSurfaces(X1_t, Y1_t, kernel, sigma=sigma)

    mm2, ss2  = gp_model.predict(XX)
    mean_zero = np.where(abs(mm2.T[0]) < 0.03)
    surf_x    = xx.T[0][mean_zero]
    surf_y    = yy.T[0][mean_zero]
    var       = np.array(ss2.T[0][mean_zero])



    ss2 = ss2.reshape(x.shape)
    z   = ss2.numpy()
    xyz = plt.pcolormesh(x, y, z, cmap=cm.Purples, vmax=z.max(), vmin=z.min())

    outer = pat.Rectangle(xy = (0.1, 0.1), width = 0.4, height = 0.4,linewidth=2, linestyle='dashdot', ec=color_cycle[5], fill=False)
    ax.add_patch(outer)

    rec2 = pat.Rectangle(xy = (0.15, 0.15), width = 0.1, height = 0.3, color=color_cycle[5])
    rec3 = pat.Rectangle(xy = (0.25, 0.15), width = 0.1, height = 0.15, color=color_cycle[5])
    rec4 = pat.Rectangle(xy = (0.35, 0.15), width = 0.1, height = 0.3, color=color_cycle[5])

    ax.add_patch(rec2)
    ax.add_patch(rec3)
    ax.add_patch(rec4)

    plt.plot(po_list[:, 0], po_list[:, 1], '--', color='black', linewidth=2, zorder=9)
    plt.scatter(po_list[-1, 0], po_list[-1, 1], c='black', s=20, marker="o",zorder=10)
    plt.scatter(surf_x, surf_y, s=3, c=color_cycle[0], zorder=8)
    set_plt()

    # plt.scatter(X2[:,0], X2[:,1])
    set_plt()
    plt.savefig('decoboco_initial.png', dpi=300, pad_inches=0.05)
    plt.show()

