#!/usr/bin/env python
# -*- coding: utf-8 -*-from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import random
import greedy_step_normal as gs
from tqdm import tqdm


"""
不確実な領域がない場合
X : data of GP
sin_list : 凹凸を判定する点

"""


# X = np.load("../data1/surf_sin_known_5000.npy")
# Y = np.zeros((X.shape[0], 1))

# sin_list = np.load("../data1/judge_point_list.npy")
# # print sin_list.shape

# num = sin_list.shape[0]

# la_list = []
# for i in tqdm(range(num)):
#     current_position = sin_list[i]
#     w = 10
#     x = np.linspace(current_position[0] - 0.005, current_position[0] + 0.005, w)
#     y = np.linspace(current_position[1] - 0.005, current_position[1] + 0.005, w)
#     z = np.linspace(current_position[2] - 0.007, current_position[2] + 0.007, w)
#     sample = np.array([x,y,z])

#     X_ = X[np.where((X[:, 0] > current_position[0]-0.008) & (X[:, 0] < current_position[0]+0.008)\
#         & (X[:, 1] > current_position[1]-0.008) & (X[:, 1] < current_position[1]+0.008))]
#     Y_ = Y[np.where((X[:, 0] > current_position[0]-0.008) & (X[:, 0] < current_position[0]+0.008)\
#         & (X[:, 1] > current_position[1]-0.008) & (X[:, 1] < current_position[1]+0.008))]

#     fp = gs.Path_Planning(X_,Y_)
#     la = fp.decision_func(sample)
#     la_list.append(la)
#     # print "la: ", la_list

# np.save("../data1/la_list_known.npy", la_list)


"""不確実な領域がある場合"""

X = np.load("../data1/surf_sin_known_5000.npy")
X_ = np.array([])
X_list = X_.tolist()
for i in range(len(X)):
    if (X[i][0] > 0.38) and (X[i][0] < 0.42) and (X[i][1] < -0.025) and (X[i][1] > -0.045):
        print "a"
    else:
        X_list.append(X[i])

X =  np.asarray(X_list)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(X[:,0], X[:,1], X[:,2], c='r', alpha=0.8, label = "correct",s=20)
# # ax.scatter(x,y,z)
# plt.show()
Y = np.zeros((X.shape[0], 1))

sin_list = np.load("../data1/judge_point_list.npy")
# # print sin_list.shape

num = sin_list.shape[0]

la_list = []
for i in tqdm(range(num)):
    current_position = sin_list[i]
    w = 10
    x = np.linspace(current_position[0] - 0.005, current_position[0] + 0.005, w)
    y = np.linspace(current_position[1] - 0.005, current_position[1] + 0.005, w)
    z = np.linspace(current_position[2] - 0.007, current_position[2] + 0.007, w)
    sample = np.array([x,y,z])

    X_ = X[np.where((X[:, 0] > current_position[0]-0.008) & (X[:, 0] < current_position[0]+0.008)\
        & (X[:, 1] > current_position[1]-0.008) & (X[:, 1] < current_position[1]+0.008))]
    Y_ = Y[np.where((X[:, 0] > current_position[0]-0.008) & (X[:, 0] < current_position[0]+0.008)\
        & (X[:, 1] > current_position[1]-0.008) & (X[:, 1] < current_position[1]+0.008))]

    fp = gs.Path_Planning(X_,Y_)
    la = fp.decision_func(sample)
    la_list.append(la)
    # print "la: ", la_list

np.save("../data1/la_list_unknown.npy", la_list)