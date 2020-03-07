#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

length=0.2
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
        # self.X = self.X[:50]
        # self.Y = self.Y[:50]

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
        mean = m + np.dot(kernel_x.T, self.b)
        # var = 1/length - np.dot(np.dot(kernel_x.T, self.invG), kernel_x)

        # if np.round((mean), 2) == 0:
        #     gp_mean = 0
        # else:
        #     gp_mean = -1

        # return gp_mean
        return mean


if __name__ == '__main__':
    # x = np.array([1,1,1])
    gpis = GPIS()
    num = 50

    mean_zero = np.array([])
    # positions = []
    # mean_zero_position = []

    x = np.linspace(0.34, 0.5, num)
    y = np.linspace(-0.06, 0.06, num)
    z = np.linspace(1.3, 1.33, num)
    # X, Y, Z = np.meshgrid(x, y, z)
    X, Y = np.meshgrid(x, y)

    # x_lim = []
    # y_lim = []
    # z_lim = []
    #
    # for i in range(num):
    #     for j in range(num):
    #         if (x[i]-0.4)**2 + y[j]**2 < 0.0025:
    #             x_lim.append(x[i])
    #             y_lim.append(y[i])
    #             z_lim.append(z[i])

    Z = np.zeros((num, num))

    count = 0
    cnt = 0
    for i in range(num):
        for j in range(num):
            for k in range(num):
                po = np.array([x[i],y[j],z[k]])

                mean_value = abs(gpis.predict_func(po))


                mean_zero = np.append(mean_zero,mean_value)
                count += 1
                print("count:", count)
                # print("mean_value", mean_value)

            print mean_zero
            n = np.argmin(mean_zero)
            print n
            print z[n]
            Z[i][j] = z[n]
            # print z_lim

            mean_zero = np.array([])


    mean_zero_position = np.array([X,Y,Z])
    print mean_zero_position

    np.save("mean_zero_1000", mean_zero_position)





            # mean_zero_position.append(positions[n])
            # positions = []
            # mean_zero = np.array([])

                # print("zero_num", cnt)

    # print np.shape(mean_zero_position)
    # # for i in range(num):
    #     for j in range(num):
    #         for k in range(num):
    #             po = np.array([X[i][j][k], Y[i][j][k], Z[i][j][k]])
    #             gpis = GPIS(po)
    #             mean_value = gpis.predict_func()
    #             if mean_value == 0:
    #                 mean_zero.append(po)


    # print(mean_zero)
    # print(X.shape)
    #

    # gpis = GPIS(x)
    # mean_value = gpis.predict_func()
    # print(mean_value)
