#!/usr/bin/env python
# -*- coding: utf-8 -*-from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


y_track = np.load("../data/unknown_y.npy")
data = np.load("../data/surf_sin_unknown2_5000.npy")
dy_track = np.load("../data/known_dy.npy")
print dy_track
# plt.plot(range(len(dy_track)), dy_track)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(y_track[:, 0], y_track[:, 1], y_track[:, 2], linewidth=3, color = "red")
ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha = 0.1, color = "blue")

# ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], alpha = 0.5, color = "blue")
# ax.scatter(self.orbit_position[:, 0], self.orbit_position[:, 1], self.orbit_position[:, 2], alpha = 0.5, color = "red")

plt.tick_params(labelsize=18)
plt.show()

# plt.savefig('../picture_movie/known_y.png')