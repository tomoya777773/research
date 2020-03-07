#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

r = 0.04
a = np.arange(0,361, 10)
b = map(lambda x :math.radians(x), a)
x = map(lambda x: 0.4 + r * math.cos(x), b)
y = map(lambda x : r * math.sin(x), b)
z = np.full(a.shape, 1.35)

circle_list = np.array([x,y,z]).T
print a.shape
print  circle_list
print(np.linalg.norm(circle_list[4] - circle_list[3]))
# fig = plt.figure()
# ax = Axes3D(fig)

# # 軸ラベルの設定
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.set_zlabel("Z-axis")

# ax.scatter(x, y, z)

# np.save("../data1/circle_r4_36.npy", circle_list)
# plt.show()
