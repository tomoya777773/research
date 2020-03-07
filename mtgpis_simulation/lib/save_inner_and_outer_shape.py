#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")

from kernel import InverseMultiquadricKernelPytouch
from mtgp import MultiTaskGaussianProcessImplicitSurfaces

import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import pickle


rate = 0.01
file_path = "../data/low_low/shape_image"

def set_plt():
    plt.xlim(0, 0.6)
    plt.ylim(0, 0.6)
    # plt.xlim(-0.2, 0.8)
    # plt.ylim(-0.2, 0.8)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.axes().set_aspect('equal', 'datalim')


## decoboco
# visual data
N1 = 10
x1_0 = (np.concatenate([np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1), np.ones(N1)*10]) ) * rate
y1_0 = (np.concatenate([np.ones(N1)*10, np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1)]) ) * rate

# true object
N2 = 10
x2_0 = np.concatenate([np.ones(N2)*15, np.linspace(15, 25, N2//2), np.ones(N2)*25, np.linspace(25, 35, N2), np.ones(N2)*35, np.linspace(35, 45, N2//2), np.ones(N2)*45, np.linspace(45, 15, N2)]) * rate
y2_0 = np.concatenate([np.linspace(15, 45, N2), np.ones(N2//2)*45, np.linspace(45, 30, N2), np.ones(N2)*30, np.linspace(30, 45, N2), np.ones(N2//2)*45, np.linspace(45, 15, N2), np.ones(N2)*15]) * rate

## high_high
# visual data
N1 = 20
x1_0 = (np.concatenate([np.linspace(15, 45, N1), np.ones(N1)*45, np.linspace(45, 15, N1), np.ones(N1)*15]) ) * rate
y1_0 = (np.concatenate([np.ones(N1)*15, np.linspace(15, 45, N1), np.ones(N1)*45, np.linspace(45, 15, N1)]) ) * rate

# true object
N2 = 10
x2_0 = (np.concatenate([np.linspace(17, 43, N2), np.ones(N2)*43, np.linspace(43, 17, N2), np.ones(N2)*17]) ) * rate
y2_0 = (np.concatenate([np.ones(N2)*17, np.linspace(17, 43, N2), np.ones(N2)*43, np.linspace(43, 17, N2)]) ) * rate

## high_low
# visual data
N1 = 10
x1_0 = (np.concatenate([np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1), np.ones(N1)*10]) ) * rate
y1_0 = (np.concatenate([np.ones(N1)*10, np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1)]) ) * rate

# true object
N2 = 10
x2_0 = (np.concatenate([np.linspace(22, 38, N2), np.ones(N2)*38, np.linspace(38, 22, N2), np.ones(N2)*22]) ) * rate
y2_0 = (np.concatenate([np.ones(N2)*30, np.linspace(30, 48, N2), np.ones(N2)*48, np.linspace(48, 30, N2)]) ) * rate


## low_high
# visual data
N1 = 10
x1_0 = (np.concatenate([np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1), np.ones(N1)*10]) ) * rate
y1_0 = (np.concatenate([np.ones(N1)*10, np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1)]) ) * rate

# true object
N2 = 10
x2_0 = (np.concatenate([np.linspace(12, 28, N2), np.ones(N2)*28, np.linspace(28, 12, N2), np.ones(N2)*12]) ) * rate
y2_0 = (np.concatenate([np.ones(N2)*12, np.linspace(12, 38, N2), np.ones(N2)*38, np.linspace(38, 12, N2)]) ) * rate

# low_low
# visual data
N1 = 10
x1_0 = (np.concatenate([np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1), np.ones(N1)*10]) ) * rate
y1_0 = (np.concatenate([np.ones(N1)*10, np.linspace(10, 50, N1), np.ones(N1)*50, np.linspace(50, 10, N1)]) ) * rate

# true object
N2 = 10
x2_0 = (np.concatenate([np.linspace(25, 35, N2), np.ones(N2)*35, np.linspace(35, 25, N2), np.ones(N2)*25]) ) * rate
y2_0 = (np.concatenate([np.ones(N2)*25, np.linspace(25, 35, N2), np.ones(N2)*35, np.linspace(35, 25, N2)]) ) * rate



fig = plt.figure(figsize=(3.0, 3.0), dpi=300)
ax = fig.add_subplot(111)

set_plt()
plt.plot(x1_0, y1_0, linewidth=3, color=(1.0, 0.0, 0.0))
plt.savefig('{}/outer_shape.png'.format(file_path), dpi=300, pad_inches=0.05)

# plt.plot(x2_0, y2_0, linewidth=3, color=(1.0, 0.0, 0.0))
# plt.savefig('{}/inner_shape.png'.format(file_path), dpi=300, pad_inches=0.05)

plt.show()