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

N1 = 32
N2 = 20
rate = 1 # data scale



x1 = (np.concatenate([np.linspace(10, 50, N1//4), np.ones(N1//4)*50, np.linspace(50, 10, N1//4), np.ones(N1//4)*10, np.array([30])])[:,None] + np.random.randn(N1+1)[:,None] * 0.2) * rate
y1 = (np.concatenate([np.ones(N1//4)*10, np.linspace(10, 50, N1//4), np.ones(N1//4)*50, np.linspace(50, 10, N1//4), np.array([30])])[:,None] + np.random.randn(N1+1)[:,None] * 0.2) * rate
X1 = np.concatenate([x1,y1], 1)
Y1 = np.concatenate([np.zeros(N1), np.array([-1])])[:, None]
T1 = 0
print(X1)
print(Y1)
x2 = (np.concatenate([np.linspace(12, 40, N2//4), np.ones(N2//4)*40, np.linspace(40, 12, N2//4), np.ones(N2//4)*12])[:,None] + np.random.randn(N2)[:,None] * 0.2) * rate
y2 = (np.concatenate([np.ones(N2//4)*12, np.linspace(12, 40, N2//4), np.ones(N2//4)*40, np.linspace(40, 12, N2//4)])[:,None] + np.random.randn(N2)[:,None] * 0.2) * rate
X2 = np.concatenate([x2,y2], 1)
Y2 = np.zeros(N2)[:, None]
T2 = 1

# X2 = np.append(X2, X1, axis=0)
# Y2 = np.append(Y2, np.ones(N1)[:, None], axis=0)

# X2 = np.append(X2, np.array([[0, 0], [10, 8], [20, 8]]), axis=0)
# Y2 = np.append(Y2, np.array([[1],[1],[1]]), axis=0)

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1.ravel(), y1.ravel(), np.array(Y1).ravel())
ax.scatter(x2.ravel(), y2.ravel(), np.array(Y2).ravel())

surf = ax.plot_surface(x, y, z, cmap='bwr', linewidth=0)
fig.colorbar(surf)
# plt.savefig('mtgpis_3d.png')
plt.show()






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

