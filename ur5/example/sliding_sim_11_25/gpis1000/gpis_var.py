#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

length=0.02
m=1
siguma=0.3


class GPIS:
    def __init__(self):
        self.X = np.load("X.npy")
        self.Y = np.load("Y.npy")
        self.X = np.delete(self.X, -1, 0)
        # print self.X.T[0]
        # print max(self.X.T[2])
        # print min(self.X.T[2])

        # self.x = np.array([X[X.shape[0]-1]]).T
        self.zeros = np.zeros((self.X.shape[0],self. X.shape[0]))
        # print(self.X.shape)
        # print(np.shape(self.X))
        # print(np.shape(self.Y))
        """カーネル行列の作成"""
        Kernel_x = np.zeros((self.X.shape[0], self.X.shape[0]))
        for i in range(self.zeros.shape[0]):
            for j in range(self.zeros.shape[0]):
                Kernel_x[i][j] = self.kernel_func(self.X[i], self.X[j])  #K

        # print Kernel_x
        G = Kernel_x + siguma*siguma * np.identity(self.zeros.shape[0])
        self.invG = np.linalg.inv(G)

        self.b = np.dot(self.invG, self.Y - m)



    def kernel_func(self, x1, x2): #kernel
        norumu2 = np.linalg.norm(x1-x2)
        kernel = pow(norumu2*norumu2 + length*length, -1/2)
        return kernel

    def predict_func(self,x):
        self.x = x

        """カーネルとカーネルの微分の列ベクト作成"""
        kernel_x = np.zeros((self.X.shape[0]))
        for i in range(self.zeros.shape[0]):
            kernel_x[i] = self.kernel_func(self.X[i], self.x)  #k(x)
        kernel_x = np.array([kernel_x]).T
        # print "kernel:", kernel_x

        """平均と分散"""
        # mean = m + np.dot(kernel_x.T, self.b)
        var = 1/length - np.dot(np.dot(kernel_x.T, self.invG), kernel_x)

        return var[0][0]

if __name__ == '__main__':
    # x = np.array([1,1,1])
    gpis = GPIS()
    X_ = np.load("mean_zero_1000.npy")

    print np.shape(X_)
    print len(X_[0])

    var_value = np.zeros((len(X_[0]),len(X_[0])))
    print var_value
    for i in range(len(X_[0])):
        for j in range(len(X_[0])):
            var_value[i][j] = gpis.predict_func([X_[0][i][j],X_[1][i][j], X_[2][i][j]])
        # var_value.append(gpis.predict_func(X_[i]))
    print(var_value)


    np.save("gp_var_1000_1", var_value)
