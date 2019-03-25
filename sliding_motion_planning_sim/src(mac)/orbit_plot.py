#!/usr/bin/env python
# -*- coding: utf-8 -*-from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

orbit_position = np.load("../data1/orbit_position_unknown.npy")
print orbit_position

plt.plot(orbit_position[:,0], orbit_position[:,1])
plt.plot(orbit_position[0][0], orbit_position[0][1], c="red")

plt.show()