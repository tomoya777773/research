#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

class GaussianProcessImplicitSurfaces:
    def __init__(self, X, Y, kernel, sigma=0.3, m=1, c=100, z_limit=0.03):
        self.X       = X
        self.Y       = Y
        self.kernel  = kernel
        self.sigma_y = sigma
        self.m       = m
        self.c       = c
        self.z_limit = z_limit

        self.KX   = self.kernel(self.X, self.X)
        L         = np.linalg.cholesky(self.KX + np.eye(self.X.shape[0]) * self.sigma_y**2)
        self.invL = np.linalg.solve(L, np.eye(L.shape[0]))
        self.invK = np.dot(self.invL.T, self.invL)

    def predict(self, x):
        # Kx  = np.diag(self.kernel(x, x))[:, None]
        Kx  = ( np.ones(len(x)) * (1/float(self.kernel.params[0])) )[:, None]
        KXx = self.kernel(self.X, x)

        # sum ( A * B.T, axis=0 ) = diag( A.dot(B) )
        invL_KXx     = self.invL.dot(KXx)
        sum_invL_KXx =  np.sum(invL_KXx * invL_KXx, axis = 0)[:, None]

        mean = self.m + KXx.T.dot(self.invK).dot(self.Y - self.m)
        cov  = Kx - sum_invL_KXx + self.sigma_y **2

        return mean, cov

    def predict_direction(self, x):
        diff_Kx = self.kernel.derivative(x, self.X)[0]
        KXx     = self.kernel(self.X, x)

        diff_mean  = diff_Kx.T.dot(self.invK.dot(self.Y - self.m))
        diff_cov   = -2 * (KXx.T.dot(self.invK).dot(diff_Kx)).T

        if x[0][2] < self.z_limit:
            diff_penalty = np.array([0, 0, -2 * self.c * (x[0][2] - self.z_limit)])[:, None]
        else:
            diff_penalty = np.zeros(3)[:, None]

        diff_object = diff_cov + diff_penalty

        normal     = diff_mean / np.linalg.norm(diff_mean, ord=2)
        projection = np.eye(self.X.shape[1]) - normal.dot(normal.T)

        S         = np.dot(projection, diff_object)
        direction = S / np.linalg.norm(S)

        return normal, direction