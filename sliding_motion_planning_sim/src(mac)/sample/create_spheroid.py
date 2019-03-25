#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sympy
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = sympy.Symbol('x')
y = sympy.Symbol('y')
z = sympy.Symbol('z')

circle =  ((x-0.4)/0.05)**2 + (y/0.05)**2 - 1
spheroid = ((x-0.4)/0.05)**2 + (y/0.05)**2 + ((z-1.3)/0.02)**2 - 1

print circle.subs([(x,1), (y,2)])

num = 100
x_ = np.linspace(0.35, 0.45, num)
y_ = np.linspace(-0.05, 0.05, num)
z_ = np.linspace(1.3, 1.35, num)

circle_plot = []
for i in tqdm(range(num)):
    for j in range(num):
        C = circle.subs([(x, x_[i]), (y, y_[j])])
        if C <= 0:
            circle_plot.append([x_[i], y_[j]])

# print circle_plot
circle_plot = np.array(circle_plot)

spheroid_list = []
for i in tqdm(range(len(circle_plot))):
    for j in range(num):
        S = spheroid.subs([(x, circle_plot[:,0][i]), (y, circle_plot[:,1][i]), (z, z_[j])])
        if round(S,2) == 0:
            spheroid_list.append([circle_plot[:,0][i], circle_plot[:,1][i], z_[j]])

spheroid_list = np.array(spheroid_list)

fig = plt.figure()
ax = Axes3D(fig)

# 軸ラベルの設定
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

ax.scatter(spheroid_list[:,0], spheroid_list[:,1], spheroid_list[:,2])

np.save("spheroid_list100.npy", spheroid_list)
plt.show()
