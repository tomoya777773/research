# -*- coding: utf-8 -*-
import numpy as np
import math

length=0.2
m=1
siguma=0.3
a=0.1



def kernel_func(x1, x2): #kernel
    norumu2 = np.linalg.norm(x1-x2)
    kernel = pow(norumu2*norumu2 + length*length, -1/2)
    return kernel

def diff_kernel(x1, x2):
    norumu2 = np.linalg.norm(x1-x2)
    diff_kernel = -norumu2 * pow(norumu2*norumu2 + length*length, -3/2)
    return diff_kernel

"""
x:三次元(x, y, z)の3*3行列
Y:c=−１，0の列ベクトル
"""

def kernel_x(X):
    zeros = np.zeros_like(X)
    kernel_x = zeros[0]
    for i in range(zeros.shape[0]):
        kernel_x[i] = kernel.func(X[i], X[X.shape[0]-1])  #k(x)
    return kernel_x

def diff_kernel_x(X):
    zeros = np.zeros_like(X)
    diff_kernel_x = zeros[0]
    for i in range(zeros.shape[0]):
        diff_kernel_x[i] = diff_kernel.func(X[i], X[X.shape[0]-1])  #k(x)
    return diff_kernel_x

def Kernel_x(X):
    zeros = np.zeros_like(X)
    Kernel_x = zeros
    for i in range(zeros.shape[0]):
        for j in range(zeros.shape[0]):
            Kernel_x[i][j] = kernel.func(X[i], X[j])  #K
    return Kernel_x

def invG(X):
    zeros = np.zeros_like(X)
    G = Kernel_x(X) + siguma*siguma * np.idetity(zeros.shape[0])
    invG = np.linalg.inv(G)
    return invG

def b(X, Y):
    b = invG(X) * (Y - m)
    return b

def mean_func(X, Y):
    mean = m + kernel_x(X).T * b(X, Y)
    return mean

def var_func(X):
    var = 1/length - kernel_x(X).T * invG(X) * kernel_x(X)
    return var

def diff_var_func(X):
    diff_var = (1/length - 2 * kernel_x(X).T * invG(X) * diff_kernel_x(X)).T
    # print 'aaa'
    return diff_var

def normal_func(X, Y):
    diff_mean = (b(X, Y).T * diff_kernel_x(X)).T
    normal = diff_mean / np.linalg.norm(diff_mean)
    return normal

def projection_func(X, Y):
    projection_n = np.identity(3) - normal_func(X, Y)*normal_func(X, Y).T
    return projection_n

def direction_func(x, Y):
    S = projection_func(x, Y) * diff_var(x, Y)
    new_direction = a * S / np.linalg.norm(S)
    return new_direction

def new_position_func(X, Y):
    x = X[X.shape[0]-1]
    new_position = x + projection_func(x, Y) * direction_func(x, Y)
    return new_position

if __name__=='__main__':
    X = np.array([[1,2,3],[2,3,2]])
    Y = np.array([0, -1])
    new_position = new_position_func(X, Y)
    print new_position
