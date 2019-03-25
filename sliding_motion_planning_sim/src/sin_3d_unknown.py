#!/usr/bin/env python
# -*- coding: utf-8 -*-from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import random
import greedy_step_6 as gs
from tqdm import tqdm

X = np.load("../data1/surf_sin_random_5000.npy")
X_ = np.array([])
X_list = X_.tolist()
for i in range(len(X)):
    if (X[i][0] > 0.38) and (X[i][0] < 0.42) and (X[i][1] < -0.025) and (X[i][1] > -0.045):
        print "a"
    else:
        X_list.append(X[i])

X =  np.asarray(X_list)

np.save("../data1/surf_sin_unknown_5000.npy", X)

