#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
y = np.array([-1,-3,-1,9,21,30,37,39,67,65,95,123,142,173,191,216,256,292,328,358])
z = np.array([-1,-3,-1,9,21,30,37,39,67,65,95,123,142,173,191,216,256,292,328,358])


def func(param,x,y,z):
    residual = z - (param[0]*x**2 + param[1]*y**2 + param[2]*x + param[3]*y + param[4])
    return residual

param = [0, 0, 0, 0, 0]
model =  optimize.leastsq(func, param, args=(x, y, z))[0]

print(model)
