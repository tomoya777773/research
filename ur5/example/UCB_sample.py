#coding: utf-8
import numpy as np
import math
import sympy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

length=0.2
m=1
siguma=0.3
a=0.03

class Find_position:
    def __init__(self, X, Y, x):
        self.X = X
        self.Y = Y
        self.x = x
        # self.x = np.array([X[X.shape[0]-1]]).T
        self.zeros = np.zeros((X.shape[0],X.shape[0]))
        # print(self.Y)

    def kernel_func(self, x1, x2): #kernel
        norumu2 = np.linalg.norm(x1-x2)
        kernel = pow(norumu2*norumu2 + length*length, -1/2)
        return kernel

    def diff_kernel_func(self, x1, x2, x3, x4):
        norumu2 = np.linalg.norm(x1-x2)
        diff_kernel = -(x3 - x4) * pow(norumu2*norumu2 + length*length, -3/2)
        return diff_kernel

    def gp_func(self):

        """カーネルとカーネルの微分の列ベクトル作成"""
        kernel_x = np.zeros((X.shape[0]))
        for i in range(self.zeros.shape[0]):
            kernel_x[i] = self.kernel_func(self.x, self.X[i])  #k(x)
        kernel_x = np.array([kernel_x]).T
        # print("kernel:", kernel_x)


        """カーネル行列の作成"""
        Kernel_x = np.zeros((X.shape[0],X.shape[0]))
        for i in range(self.zeros.shape[0]):
            for j in range(self.zeros.shape[0]):
                Kernel_x[i][j] = self.kernel_func(self.X[i], self.X[j])  #K

        # print(Kernel_x)
        G = Kernel_x + siguma*siguma * np.identity(self.zeros.shape[0])
        invG = np.linalg.inv(G)

        b = np.dot(invG, self.Y - m)
        # print("b:", b)

        """平均と分散"""
        mean = m + np.dot(kernel_x.T, b)
        var = 1/length - np.dot(np.dot(kernel_x.T, invG), kernel_x)
        return mean, var

# class Integral_func:
#     def __init__():


if __name__=='__main__':
    print '##################'
    X = np.array([[0.9, 0.1, math.sqrt(0.19)],[0.8999, 0.43, 0.1]])
    # X = np.array([[0.1,math.sqrt(0.99)],[0.11, 0.99]])

    Y = np.array([[0],[-1]])

    x_sample = np.arange(-1.1, 1.1, 0.04)
    y_sample = np.arange(-1.1, 1.1, 0.04)
    z_sample = np.arange(-1.1, 1.1, 0.04)
    print x_sample
    xy_sample = []
    X_sample = []
    Y_sample = []
    Z_sample = []
    for i in range(len(x_sample)):
        for j in range(len(y_sample)):
            for k in  range(len(z_sample)):
                fp = Find_position(X, Y, [x_sample[i], y_sample[j], z_sample[k]])
                pred = fp.gp_func()
                # print(pred[0][0][0])
                if round(pred[0][0][0], 2) == 0:
                    xy_sample.append([x_sample[i],y_sample[j], z_sample[k]])
                    X_sample.append(np.round(x_sample[i],5))
                    Y_sample.append(np.round(y_sample[j],5))
                    Z_sample.append(np.round(z_sample[k],5))
                    # print pred
    # print xy_sample
    print X_sample
    print Y_sample

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_sample, Y_sample, Z_sample)
    # plt.scatter(X_sample, Y_sample, Z_sample)
    plt.show()


    # fig.show()
    # fp = Find_position(X, Y, [0.87,0.48])
    # pred = fp.gp_func()
    # print(pred)
