#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sympy import *


class GaussianProcessImplicitSurface:

    """
    Attributes
    ----------
    X : float
        Observation data (x,y,z)
    Y : -1, 0, 1
        Label to determine contact in observation data

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
                 length=0.2, sigma=0.01, c=100, z_limit=1.25, a = 0.03):

        print "----------CREATE GPIS----------"

        self.X = position_data
        self.Y = label_data
        num = self.X.shape[0]

        self.m = m
        self.length = length
        self.c = c
        self.z_limit = z_limit
        self.a = a

        """Create kernel matrix"""

        self.invG = self.calculate_invesrse(self.X, self.length, sigma)
        self.b = np.dot(self.invG, self.Y - self.m)

        # Kernel_x = np.zeros((num, num))
        # for i in range(num):
        #     for j in range(num):
        #         Kernel_x[i][j] = self.kernel_func(self.X[i], self.X[j])  #K
        # Kernel_x = np.array(Kernel_x)
        # # print"XXXXXXXXXxx", self.X.shape
        # """G, b"""
        # G = Kernel_x + sigma**2 * np.identity(num)
        # self.invG = np.linalg.inv(G)
        # self.b = np.dot(self.invG, self.Y - self.m)

        # print "Kernel_x:", Kernel_x
        # print "G:" , G
        # print "invG:", self.invG
        # print "b:", self.b


    def calculate_invesrse(self, X, l, sigma):
        sum_X_2 = np.sum(X**2, axis=1).reshape(-1,1)
        K = np.power(sum_X_2 - 2*np.dot(X,X.T) + sum_X_2.T + l**2, -1/2)
        G = K + sigma**2 * np.identity(X.shape[0])
        L = np.linalg.cholesky(G)
        inverse_L = np.linalg.solve(L,np.identity(X.shape[0]))
        inverse_G = np.dot(inverse_L.T, inverse_L)
        return inverse_G

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

    """normal and tangent"""
    def direction_func(self, current_position, orbit_position=None, w_param=[0, 1, 1], data_number=100):

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

        a = w_param[0]
        b = w_param[1]
        c = w_param[2]

        if data_number < 25:
            a,b = 1,1
        # elif 10 <= data_number <= 20:
        #     a,b = 6,2
        # elif 20 < data_number < 30:
        #     a,b = 4,6

        """kernel and diff kernel"""
        self.calculate_kernel(current_position)
        self.calculate_diff_kernel(current_position)

        """normal"""
        diff_mean = (np.dot(self.b.T, self.diff_kernel)).T
        normal = diff_mean / np.linalg.norm(diff_mean, ord=2)

        """projection"""
        projection_n = np.identity(self.X.shape[1]) - np.dot(normal, normal.T)

        """diff variance"""
        diff_var =  (- 2 * np.dot(np.dot(self.kernel, self.invG), self.diff_kernel)).T
        diff_var =  -diff_var / np.linalg.norm(diff_var)

        """diff orbit"""
        if orbit_position is not None:
            orbit_position_copy = np.copy(orbit_position)
            orbit_position_copy[2] = current_position[2]
            diff_orbit = (orbit_position_copy - current_position) / np.linalg.norm(current_position - orbit_position_copy, ord=2)
        else:
            diff_orbit = np.zeros(current_position.shape[0])

        """diff penalty"""
        if current_position[-1] < self.z_limit and current_position.shape[0] > 2:
            diff_penalty = np.array([0, 0, -2 * self.c * (current_position[-1] - self.z_limit)])
            diff_penalty = diff_penalty / np.linalg.norm(diff_penalty)
        else:
            diff_penalty = np.zeros(current_position.shape[0])

        """diff object function"""
        diff_object = a * diff_var + b * diff_orbit + c * diff_penalty
        # diff_object /= np.linalg.norm(diff_orbit)
        # diff_object = diff_var

        """tangent direction"""
        S = np.dot(projection_n, diff_object)
        direction = S / np.linalg.norm(S, ord=2)

        # print self.b
        # print("normal:", normal)
        # print projection_n
        # print "diff_var: ", diff_var
        # print "diff_orbit: ", diff_orbit
        # print "diff_penalty: ", diff_penalty
        # print "diff_object:", diff_object

        return normal.T[0], direction
