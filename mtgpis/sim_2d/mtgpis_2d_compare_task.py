#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")

from kernel import InverseMultiquadricKernelPytouch
from mtgp import MultiTaskGaussianProcessImplicitSurfaces

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.style.use("ggplot")

N1 = 60
N2 = 27
rate = 1 # data scale

# x1 = (np.concatenate([np.linspace(0, 7, N1//5), np.ones(N1//5)*8, np.linspace(9, 41, N1//5), np.ones(N1//5)*42, np.linspace(43, 50, N1//5)])[:,None] + np.random.randn(N1)[:,None] * 0.2) * rate
# y1 = (np.concatenate([np.ones(N1//5)*40, np.linspace(40, 62, N1//5), np.ones(N1//5)*62, np.linspace(62, 40, N1//5), np.ones(N1//5)*40])[:,None] + np.random.randn(N1)[:,None] * 0.2) * rate
# X1 = np.concatenate([x1,y1], 1)
# Y1 = np.zeros(N1)[:, None]
# T1 = 0

x1 = (np.concatenate([np.linspace(0, 7, N1//5), np.ones(N1//5)*8, np.linspace(9, 41, N1//5), np.ones(N1//5)*42, np.linspace(43, 50, N1//5)])[:,None] + np.random.randn(N1)[:,None] * 0.2) * rate
y1 = (np.concatenate([np.ones(N1//5)*2, np.linspace(2, 32, N1//5), np.ones(N1//5)*32, np.linspace(32, 2, N1//5), np.ones(N1//5)*2])[:,None] + np.random.randn(N1)[:,None] * 0.2) * rate
X1 = np.concatenate([x1,y1], 1)
Y1 = np.zeros(N1)[:, None]
T1 = 0

x2 = (np.concatenate([np.linspace(0, 10, N2//9), np.ones(N2//9)*10, np.linspace(10, 20, N2//9), np.ones(N2//9)*20, np.linspace(20, 30, N2//9), np.ones(N2//9)*30, np.linspace(30, 40, N2//9), np.ones(N2//9)*40, np.linspace(40, 50, N2//9)])[:,None] + np.random.randn(N2)[:,None] * 0.2) * rate
y2 = (np.concatenate([np.zeros(N2//9), np.linspace(0, 30, N2//9), np.ones(N2//9)*30, np.linspace(30, 0, N2//9), np.zeros(N2//9), np.linspace(0, 30, N2//9), np.ones(N2//9)*30, np.linspace(30, 0, N2//9), np.zeros(N2//9)])[:,None] + np.random.randn(N2)[:,None] * 0.2) * rate
X2 = np.concatenate([x2,y2], 1)
Y2 = np.zeros(N2)[:, None]
T2 = 1


# x2 = (np.concatenate([np.linspace(0, 10, N2//5), np.ones(N2//5)*11, np.linspace(11, 39, N2//5), np.ones(N2//5)*39, np.linspace(39, 50, N2//5)])[:,None] + np.random.randn(N2)[:,None] * 0.2) * rate
# y2 = (np.concatenate([np.zeros(N2//5), np.linspace(0, 30, N2//5), np.ones(N2//5)*30, np.linspace(30, 0, N2//5), np.zeros(N2//5)])[:,None] + np.random.randn(N2)[:,None] * 0.2) * rate
# X2 = np.concatenate([x2,y2], 1)
# Y2 = np.zeros(N2)[:, None]
# T2 = 1

# N2 = 40
# x2 = (np.concatenate([np.linspace(0, 10, N2//5), np.ones(N2//5)*10, np.linspace(10, 20, N2//5), np.ones(N2//5)*20, np.linspace(20, 30, N2//5)])[:,None] + np.random.randn(N2)[:,None] * 0.2) * rate
# y2 = (np.concatenate([np.zeros(N2//5), np.linspace(0, 30, N2//5), np.ones(N2//5)*30, np.linspace(30, 0, N2//5), np.ones(N2//5)])[:,None] + np.random.randn(N2)[:,None] * 0.2) * rate
# X2 = np.concatenate([x2,y2], 1)
# Y2 = np.zeros(N2)[:, None]
# T2 = 1

# X2 = np.append(X2, X1, axis=0)
# Y2 = np.append(Y2, np.ones(N1)[:, None], axis=0)
# X1 = np.append(X1, np.array([[21, 30], [22, 32], [25, 3], [28, 2], [22, 15], [24, 13], [27, 13], [23, 25], [26, 26]]), axis=0)
# Y1 = np.append(Y1, np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1]]), axis=0)*(-1)

# X1 = np.append(X1, np.array([[22, 4], [28, 15]]), axis=0)
# Y1 = np.append(Y1, np.array([[-1], [-1]]), axis=0)
X2 = np.append(X2, np.array([[24, 20], [26, 22], [28, 2]]), axis=0)
Y2 = np.append(Y2, np.array([[1],[1],[1]]), axis=0)
# X1 = np.append(X1, X2, axis=0)
# Y1 = np.append(Y1, np.ones(N2)[:, None]*(-1), axis=0)

# X1 = np.append(X1, np.concatenate([np.ones(N2//4)[:,None]*20, np.linspace(30, 0, N2//4)[:,None]], 1), axis=0)
# Y1 = np.append(Y1, np.ones(N2//4)[:, None]*(-1), axis=0)

# X3 = X2 + np.array([0, 1])
# X2 = np.append(X2, X3, axis=0)
# Y2 = np.append(Y2, np.ones(N2)[:, None], axis=0)

# x3 = np.concatenate([np.ones(N2//9)*20, np.linspace(20, 30, N2//9), np.ones(N2//9)*30])[:, None]
# y3 = np.concatenate([np.linspace(30, 0, N2//9), np.zeros(N2//9), np.linspace(0, 30, N2//9)])[:, None]
# X3 = np.concatenate([x3,y3], 1)
# Y3 = np.ones(N2//9 * 3)[:, None] * (-1)
# X1 = np.append(X1, X3, axis=0)
# Y1 = np.append(Y1, Y3, axis=0)

X1 = torch.from_numpy(X1).float()
X2 = torch.from_numpy(X2).float()
Y1 = torch.from_numpy(Y1).float()
Y2 = torch.from_numpy(Y2).float()

kernel = InverseMultiquadricKernelPytouch([0.4])
model  = MultiTaskGaussianProcessImplicitSurfaces([X1,X2], [Y1,Y2], [T1,T2], kernel)

x   = np.linspace(-10, 60, 100)[:, None] * rate
y   = np.linspace(-20, 70, 100)[:, None] * rate

# x   = np.linspace(0, np.pi*4, 100)[:, None] * rate
# y   = np.linspace(-2, 5, 100)[:, None] * rate

x,y = np.meshgrid(x, y)
xx  = x.ravel()[:, None]
yy  = y.ravel()[:, None]
XX  = np.concatenate([xx, yy], 1)
XX  = torch.from_numpy(XX).float()


mm1, ss1 = model.predict(XX, 0)
mm2, ss2 = model.predict(XX, 1)

mm1 = mm1.reshape(x.shape)
mm2 = mm2.reshape(x.shape)

z = mm2.numpy()
# xyz = plt.pcolormesh(x, y, z, cmap="Greens", shading="gouraud", vmax=z.max(), vmin=z.min())
plt.scatter(x1.ravel(), y1.ravel())
plt.scatter(x2.ravel(), y2.ravel())
# plt.axis([x.min(), x.max(), y.min(), y.max()])
# plt.colorbar(xyz)
# plt.savefig('env1.png')
plt.show()


############################################
print("----- before learning -----")
print(model.task_params_to_psd())
model.learning(max_iter=5000)
print("----- after learning -----")
print(model.task_params_to_psd())
############################################
# 相関行列のヒートマップを描く
colormap = plt.cm.RdBu

sns.heatmap(model.task_params_to_psd().numpy(), linewidths=0.1, 
        square=True, cmap=colormap, linecolor='white')
plt.savefig('similarity.png')

# グラフを表示する
plt.show()

mm1, ss1 = model.predict(XX, 0)
mm2, ss2 = model.predict(XX, 1)

mm1 = mm1.reshape(x.shape)
mm2 = mm2.reshape(x.shape)

z = mm2.numpy()

xyz = plt.pcolormesh(x, y, z, cmap='Greens', shading="gouraud", vmax=z.max(), vmin=z.min())
plt.scatter(x1.ravel(), y1.ravel())
plt.scatter(x2.ravel(), y2.ravel())
plt.colorbar(xyz)
# plt.savefig('mtgpis.png')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x1.ravel(), y1.ravel(), np.array(Y1).ravel())
# ax.scatter(x2.ravel(), y2.ravel(), np.array(Y2).ravel())

# surf = ax.plot_surface(x, y, np.array(mm2), cmap='bwr', linewidth=0)
# fig.colorbar(surf)
# # plt.savefig('mtgpis_3d.png')
# plt.show()






#### 2d -> 3dsurface
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x1.ravel(), y1.ravel(), np.array(Y1).ravel())
# ax.scatter(x2.ravel(), y2.ravel(), np.array(Y2).ravel())

# print(x.shape)
# print(mm2.shape)
# surf = ax.plot_surface(x, y, np.array(mm2), cmap='bwr', linewidth=0)
# fig.colorbar(surf)
# plt.show()

