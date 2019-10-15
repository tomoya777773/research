#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm, multivariate_normal


"""
現在位置が物体上か外か中か判定する
"""

def judge_ellipse(position, r):


    """
    Paraneters
    ----------
    position :  1d array
    r : radius

    Returns
    -------
    -1 : inside
    0  : on
    1  : outside

    """

    x,y,z = position
    judge = (x**2 / 4 + y**2 / 4 + z**2) - r**2
    if judge < 0: # in
        return -1
    elif judge > 0 and round(judge, 4) == 0: # on
        return 0
    else: # out
        return 1



def func(x, y):
    m=2
    mean = np.zeros(m)
    sigma = np.eye(m) * 30
    X = np.c_[np.ravel(x), np.ravel(y)] * 30

    Y_plot = 100 * (multivariate_normal.pdf(x=X, mean=mean, cov=sigma) - 0.35*multivariate_normal.pdf(x=X, mean=mean, cov=sigma*0.5))-0.045

    # Y_plot = Y_plot.reshape(x.shape)

    return Y_plot

def judge_object1(position):
    x,y,z = position
    judge = z - func(x,y)
    # print judge
    if judge < 0: # in
        return -1
    elif judge > 0 and round(judge, 4) == 0: # on
        return 0
    else: # out
        return 1