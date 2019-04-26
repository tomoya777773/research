#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sympy import *


class GaussianProcessImplicitSurface:

    """
    1　GPの平均と分散
    2　凹凸判定
    3　経路決定

    Attributes
    ----------
    X : float
        観測点(x,y,z)
    Y : -1, 0, 1
        観測点の接触を判断するラベル

    Notes
    ----------
    kernel_x : vector
        inverse-multiquadric kernel
    diff_kernel_x : matrix
        differential kernel_x
    Kernel_x : matrix
        kernel function
    """

    def __init__(self, position_data, label_data, m=1,
                 length=0.2, sigma=0.01, c=100, z_limit=1.28, a = 0.03):

        """
        Parameters
        ----------
        X : float
            観測点(x,y,z)
        Y : -1, 0, 1
            観測点の接触を判断するラベル
        """

        print "----------CREATE GPIS----------"

        self.X = position_data
        self.Y = label_data
        num = self.X.shape[0]

        self.m = m
        self.length = length
        self.c = c
        self.z_limit = z_limit
        self.a = a

        """カーネル行列作成"""
        Kernel_x = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                Kernel_x[i][j] = self.kernel_func(self.X[i], self.X[j])  #K
        Kernel_x = np.array(Kernel_x)
        # print "kkkkkk:", Kernel_x

        """G, b"""
        G = Kernel_x + sigma**2 * np.identity(num)
        self.invG = np.linalg.inv(G)
        self.b = np.dot(self.invG, self.Y - self.m)
        # print "G:" , G
        # print "b:", self.b

    def kernel_func(self, data_position, current_position):
        kernel = pow(np.linalg.norm(current_position - data_position, ord=2)**2 + self.length**2 , -1/2)
        return kernel

    def diff_kernel_func(self, data_position, current_position):
        diff_kernel = - pow(np.linalg.norm(current_position - data_position, ord=2)**2 + self.length**2, -3/2) * (current_position - data_position)
        return diff_kernel

    def calculate_kernel(self, current_position):
        kernel = [self.kernel_func(self.X[i], current_position) for i in range(self.X.shape[0])]
        self.kernel = np.array(kernel)

    def calculate_diff_kernel(self, current_position):
        diff_kernel = [self.diff_kernel_func(self.X[i], current_position) for i in range(self.X.shape[0])]
        self.diff_kernel = np.array(diff_kernel)

    def calcurate_mean(self):
        return self.m + np.dot(self.kernel, self.b)

    def calcurate_variance(self):
        return 1/self.length - np.dot(np.dot(self.kernel,self. invG), self.kernel)

    """凹凸判定"""
    def decision_func(self, po_list, x_a):

        """
        Parameters
        ----------
        x, y, z : float
            凹凸を判定する位置

        Returns
        ----------
        CC : 0 or 1 (0 : 凹, 1 : 凸)
            convexoconcave
        """
        x, y = po_list[0],  po_list[1]
        position_list, mean_list, mean_zero_list, var, var_list = [],[],[],[],[]
        num = x.shape[0]
        for i in range(num):
            for j in range(num):
                po = [x[i],y[j]]
                position_list.append(po)
                self.calculate_kernel(po)

                mean_list.append(abs(self.calcurate_mean()))
                var.append(self.calcurate_variance())

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

    """法線方向"""
    def direction_func(self, current_position):

        """kernel_x_
        Parameters
        ----------

        Returns
        ----------
        normal : vector
            normal against object
        d : vector
            new direction on projection
        """
        """カーネルとカーネル微分"""
        self.calculate_kernel(current_position)
        self.calculate_diff_kernel(current_position)

        """normal"""
        diff_mean = (np.dot(self.b.T, self.diff_kernel)).T
        normal = diff_mean / np.linalg.norm(diff_mean, ord=2)

        # print self.b
        # print("normal:", normal)

        """projection"""
        projection_n = np.identity(self.X.shape[1]) - np.dot(normal, normal.T)

        # print projection_n
        """分散の微分"""
        diff_var =  (- 2 * np.dot(np.dot(self.kernel, self.invG), self.diff_kernel)).T
        diff_var = - diff_var / np.linalg.norm(diff_var)
        # print "diff_var: ", diff_var

        """新しい方向"""
        S = np.dot(projection_n, diff_var)
        direction = self.a * S / np.linalg.norm(S, ord=2)

        return normal.T[0], direction
