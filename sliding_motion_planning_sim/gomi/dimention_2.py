#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import *

length=0.2
m=1
siguma=0.3
a=0.03

class Find_position:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        num = X.shape[0]

        """カーネル関数とカーネルの微分関数を定義"""
        kernel_func = lambda a,b : pow(np.linalg.norm(a-b)**2 + length**2 , -1/2)
        diff_kernel_func = lambda a,b,c,d : -(c-d) * pow(np.linalg.norm(a-b)**2 + length**2 , -3/2)

        """カーネルベクトル作成"""
        kernel_x = [kernel_func(self.X[i], self.X[num -1]) for i in range(num)]
        self.kernel_x = np.array(kernel_x)
        # print("kernel:", kernel_x)

        """カーネルの微分ベクトル作成"""
        self.diff_kernel_x = np.zeros((num, self.X.shape[1]))
        for i in range(num):
            for j in range(self.X.shape[1]):
                self.diff_kernel_x[i][j] = diff_kernel_func(self.X[num-1], self.X[i], self.X[num-1][j], self.X[i][j])
        self.diff_kernel_x = np.array(self.diff_kernel_x)
        # print("diff_kernel:", self.diff_kernel_x)

        """カーネル行列作成"""
        self.Kernel_x = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                self.Kernel_x[i][j] = kernel_func(self.X[i], self.X[j])  #K
        self.Kernel_x = np.array(self.Kernel_x)
        # print(self.Kernel_x)

        """G, bを求める"""
        G = self.Kernel_x + siguma**2 * np.identity(num)
        self.invG = np.linalg.inv(G)
        self.b = np.dot(self.invG, self.Y - m)
        # print("b:", self.b)

    """平均と分散"""
    def mean_and_var(self, x):
        kernel_func = lambda a,b : pow(np.linalg.norm(a-b)**2 + length**2 , -1/2)
        kernel_x_ = [kernel_func(self.X[i], x) for i in range(self.X.shape[0])]
        self.kernel_x_ = np.array(kernel_x_)
        # print("self.kernel_x:", self.kernel_x_)
        mean = m + np.dot(self.kernel_x_, self.b)
        var = 1/length - np.dot(np.dot(self.kernel_x_,self. invG), self.kernel_x_)

        return mean, var

    """固有値計算"""
    def decision_func(self, x, y, x_a):

        position_list, mean_list, mean_zero_list, var, var_list = [],[],[],[],[]
        num = 50
        for i in range(num):
                for j in range(num):
                        po = [x[i],y[j]]
                        position_list.append(po)
                        mean_var = self.mean_and_var(po)
                        mean_list.append(abs(mean_var[0]))
                        var.append(mean_var[1])

                m_list = np.array(mean_list)
                n = m_list.argmin()

                mean_zero_list.append(position_list[n])
                var_list.append(var[n])

                position_list, mean_list, var = [],[],[]

        mean_zero_list = np.array(mean_zero_list)
        # var_list = np.reciprocal(var_list)

        mean_x = mean_zero_list[::, 0]
        mean_y = mean_zero_list[::, 1]

        z = np.polyfit(mean_x, mean_y, 2)

        x = Symbol('x')
        y = Symbol('y')

        # equ = z[0]*x**4 + z[1]*x**3 + z[2]*x**2 + z[3]*x + z[4]
        # equ = z[0]*x**3 + z[1]*x**2 + z[2]*x + z[3]
        equ = z[0]*x**2 + z[1]*x + z[2]

        equ_diff = diff(equ, x)
        equ_diff_diff = diff(equ_diff, x)
        la = equ_diff_diff.subs(x, x_a)

        return la

    """法線と方向"""
    def direction_func(self):
        print("b:", self.b)
        print("diff_k:", self.diff_kernel_x)
        self.diff_mean = (np.dot(self.b.T, self.diff_kernel_x)).T
        self.diff_var = (- 2 * np.dot(np.dot(self.kernel_x, self.invG), self.diff_kernel_x)).T
        print("diffffff:", self.diff_var)

        """法線"""
        normal = self.diff_mean / np.linalg.norm(self.diff_mean)
        print("noemal:", normal)
        """接平面"""
        projection_n = np.identity(self.X.shape[1]) - np.dot(normal, normal.T)

        """新しい方向"""
        S = np.dot(projection_n, self.diff_var)
        new_direction = S / np.linalg.norm(S)
        d = np.dot(projection_n, new_direction)

        # print("normal:",normal)
        # print("p:", projection_n)
        # print("S:", S)
        # print("new:", new_direction)
        # print("d", d)
        return normal, d

if __name__=='__main__':

    """show object"""
    x1 = np.arange(-3,3,0.01)
    y1 = np.sin(x1)
    y2 = -2
    plt.fill_between(x1,y1,y2,facecolor='g')
    plt.gca().set_aspect('equal', adjustable='box')

    """show sample plot"""

    num = 18 # the number of samples

    x_s = np.linspace(-3, 3, num)
    y_s = np.sin(x_s)

    X = np.zeros((num,2))
    X[:, 0] = x_s
    X[:, 1] = y_s

    Y = np.zeros(num)
    Y = Y[:, np.newaxis]
    print(X)

    """search"""
    n = 50

    x_ = np.linspace(-2, 2, n)
    y_ = np.sin(x_)

    true_x, true_y, false_x, false_y, la_list = [],[],[],[],[]
    count, true_count, false_count = 0,0,0

    for i in range(n):
        x = np.linspace(x_[i] - 1, x_[i] + 1, n)
        y = np.linspace(y_[i] - 1, y_[i] + 1, n)

        # print(X[np.where((X[:, 0] > x_[i]-0.5) & (X[:, 0] < x_[i]+0.5))])
        # print(Y[np.where((X[:, 0] > x_[i]-0.2) & (X[:, 0] < x_[i]+0.2))])

        X_ = X[np.where((X[:, 0] > x_[i]-1) & (X[:, 0] < x_[i]+1))]
        Y_ = Y[np.where((X[:, 0] > x_[i]-1) & (X[:, 0] < x_[i]+1))]

        fp = Find_position(X_, Y_)
        la = fp.decision_func(x, y, x_[i])

        la_list.append(la)

        if (la > 0 and y_[i] < 0) or (la <= 0 and y_[i] > 0):
            true_x.append(x_[i])
            true_y.append(y_[i])
            true_count+=1
        else:
            false_x.append(x_[i])
            false_y.append(y_[i])
            false_count+=1
        count += 1
        print("count:", count)

    # print(la_list)
    print("true count:", true_count)
    print("false count:", false_count)
    print('true accurency:', true_count/n)
    print('false accurency:', false_count/n)


    plt.scatter(true_x, true_y, marker="x", color='r', alpha=0.8, label = "correct",s=70)
    plt.scatter(false_x, false_y, marker="v", color='b', alpha=0.8, label = "incorrect",s=70)
    plt.scatter(x_s, y_s, marker='o', color='black',label="sample",s=30)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='upper left',fontsize=16)

    # plt.savefig('d2_point20.png')
    plt.show()
