#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from scipy.optimize import fmin_powell, fmin, fmin_l_bfgs_b
from scipy.linalg import cholesky, solve
# import  time

class MultiTaskGaussianProcessImplicitSurfaces:
    def __init__(self, X, Y, T, kernel, task_kernel_params, sigma=0.3, m=1, c=100, z_limit=0.03):
        self.X1 = X[0]
        self.X2 = X[1]
        self.Y1 = Y[0]
        self.Y2 = Y[1]
        self.X  = np.concatenate(X)
        self.Y  = np.concatenate(Y)
        self.T  = np.concatenate([np.ones(X[i][:, 0].shape, dtype=np.long)*T[i] for i in range(len(X))])[:, None]

        self.sigma_y = sigma
        self.m       = m
        self.c       = c
        self.z_limit = z_limit

        self.kernel = kernel

        self.task_kernel_params = task_kernel_params
        self.task_kernel  = self.task_to_psd_matrix(self.task_kernel_params)
        # self.task_kernel /= self.task_kernel[1][1]

        self.KX   = self.kernel(self.X, self.X)
        KT        = self.task_kernel[self.T, self.T.T]
        L         = np.linalg.cholesky(KT * self.KX + np.eye(self.X.shape[0]) * self.sigma_y**2)
        self.invL = np.linalg.solve(L, np.eye(L.shape[0]))
        self.invK = np.dot(self.invL.T, self.invL)


    def task_to_psd_matrix(self, task_kernel_params):
        L   = np.exp(task_kernel_params)
        res = np.dot(L, L.T)
        return res

    def calculate_inv(self, K):
        L = np.linalg.cholesky(K)
        inv_L = np.linalg.solve(L, np.eye(L.shape[0]))
        inv_K = np.dot(inv_L.T, inv_L)
        return inv_K

    def predict(self, x, t):
        t   = np.reshape(np.ones(x.shape[0], dtype=np.long) * t, (-1, 1))

        Kt  = self.task_kernel[t, t]
        KTt = self.task_kernel[self.T, t.T]

        Kx  = ( np.ones(len(x)) * (1/float(self.kernel.params[0])) )[:, None]
        KXx = self.kernel(self.X, x)

        kk  = Kt  * Kx
        k   = KTt * KXx

        # sum ( A * B.T, axis=0 ) = diag( A.dot(B) )
        invL_k = self.invL.dot(k)

        mean = self.m + k.T.dot(self.invK).dot(self.Y - self.m)
        cov  = kk - np.sum(invL_k * invL_k, axis = 0)[:, None] + self.sigma_y**2

        return mean, cov

    def log_likelihood(self, params):
        if params.ndim == 2:
            params = params[0]

        task_kernel_params = np.array( [ [ params[0], 0 ], [ params[1], params[2] ] ] )
        task_kernel        = self.task_to_psd_matrix(task_kernel_params)

        KT = task_kernel[self.T, self.T.T]

        K   = KT * self.KX + np.eye(self.X.shape[0]) * self.sigma_y**2

        L = np.linalg.cholesky(K)
        invL = np.linalg.solve(L, np.eye(L.shape[0]))
        invK = np.dot(invL.T, invL)

        Y_m = self.Y - self.m

        return  np.linalg.slogdet(K)[1] + Y_m.T.dot(invK).dot(Y_m)

    def log_likelihood_plus_kernel_params(self, params):
        if params.ndim == 2:
            params = params[0]

        task_kernel_params = np.array( [ [ params[0], 0 ], [ params[1], params[2] ] ] )
        task_kernel        = self.task_to_psd_matrix(task_kernel_params)
        self.kernel.params = np.array([params[3]])
        # print self.kernel.params

        KT = task_kernel[self.T, self.T.T]

        Eye = np.eye(self.X.shape[0]) * self.sigma_y**2
        K   = KT * self.KX + Eye
        Y_m = self.Y - self.m

        return  np.linalg.slogdet(K)[1] + Y_m.T.dot(self.calculate_inv(K)).dot(Y_m)

    def gradient(self, params):
        if params.ndim == 2:
            params = params[0]

        task_kernel_params = np.array([[params[0], 0], [params[1], params[2]]])
        task_kernel = self.task_to_psd_matrix(task_kernel_params)

        KT = task_kernel[self.T, self.T.T]

        Eye   = np.eye(self.X.shape[0]) * self.sigma_y**2
        K     = KT * self.KX + Eye
        invK  = self.calculate_inv(K)
        invKY = invK.dot(self.Y)

        # grad_l0 = np.array([ [ 2*np.exp(2*params[0])        , np.exp(params[0] + params[1]) ],\
        #                      [ np.exp(params[0] + params[1]), 0                             ]])
        # grad_l1 = np.array([ [ 0                            , np.exp(params[0] + params[1]) ],\
        #                      [ np.exp(params[0] + params[1]), 2*np.exp(2*params[1])         ]])
        # grad_l2 = np.array([ [ 0                            , 0                             ],\
        #                      [ 0                            , 2*np.exp(2*params[2])         ]])


        grad_l0 = np.array([ [ 2*np.exp(2*params[0])        , np.exp(params[0] + params[1]) ],\
                             [ np.exp(params[0] + params[1]), 0                             ]])
        grad_l1 = np.array([ [ 0                            , np.exp(params[0] + params[1]) ],\
                             [ np.exp(params[0] + params[1]), 2*np.exp(2*params[1])         ]])
        grad_l2 = np.array([ [ 0                            , 0                             ],\
                             [ 0                            , 2*np.exp(2*params[2])         ]])


        N      = len(params)
        grad_l = [grad_l0, grad_l1, grad_l2]
        grad   = np.zeros(N)

        for i in range(N):
            grad_T  = grad_l[i][self.T, self.T.T]
            grad_KT = grad_T * self.KX
            grad[i] = np.trace(invK.dot(grad_KT)) - invKY.T.dot(grad_KT).dot(invKY)

        return grad

    def learn_params(self):
        tk_params = np.array([self.task_kernel_params[0][0], self.task_kernel_params[1][0], self.task_kernel_params[1][1]])

        # print "init params:", tk_params
        res = fmin(
            func        = self.log_likelihood,
            x0          = tk_params,
            # fprime = self.gradient,
            xtol        = 1e-4,
            ftol        = 1e-4,
            maxiter     = 1000,
            maxfun      = 1000,
            retall      = True,
            full_output = True,
        )
        xopt = res[0]
        # print "##########################3"
        # res = fmin_l_bfgs_b(func, tk_params, fprime=fprime)
        # res = fmin_cg(func, tk_params, fprime=fprime, maxiter=100, full_output=True, disp=True, retall=True)
        # res = fmin_bfgs(func, tk_params, fprime=fprime, maxiter=100)
        # print "res:  ",res
        # xopt = res
        # print "xcccccccccccccccccccc", check_grad(func, fprime, np.array([5,-2,2]))

        # print xopt
        self.task_kernel_params = np.array([[xopt[0], 0], [xopt[1], xopt[2]]])

        print("task_kernel_params:", self.task_kernel_params)
        print("task_kernel:", self.task_to_psd_matrix(self.task_kernel_params))
#
    def learn_params_plus_kernel_params(self):
        tk_params = np.array([self.task_kernel_params[0][0], self.task_kernel_params[1][0], self.task_kernel_params[1][1]])
        # print tk_params
        # print self.kernel.params
        init_params = np.concatenate([tk_params, self.kernel.params])
        # print "init params:", tk_params

        res = fmin(
            func        = self.log_likelihood_plus_kernel_params,
            x0          = init_params,
            xtol        = 1e-4,
            ftol        = 1e-4,
            maxiter     = 1000,
            maxfun      = 1000,
            retall      = True,
            full_output = True,
        )
        xopt = res[0]
        # print xopt
        self.task_kernel_params = np.array([[xopt[0], 0], [xopt[1], xopt[2]]])
        self.kernel.parmas = xopt[3]

        print("task_kernel_params:", self.task_kernel_params)
        print("task_kernel:", self.task_to_psd_matrix(self.task_kernel_params))
        # print "kernel_params:", self.kernel.params

    def predict_direction(self, x, t):
        t   = np.reshape(np.ones(x.shape[0], dtype=np.long) * t, (-1, 1))
        KTt = self.task_kernel[self.T, t.T]

        diff_Kx = self.kernel.derivative(x, self.X)[0]
        KXx     = self.kernel(self.X, x)

        kk      = KTt * diff_Kx
        k       = KTt * KXx

        diff_mean  = kk.T.dot(self.invK).dot(self.Y - self.m)
        diff_cov   = -2 * (k.T.dot(self.invK).dot(kk)).T

        if x[0][2] < self.z_limit:
            diff_penalty = np.array([0, 0, -2 * self.c * (x[0][2] - self.z_limit)])[:, None]
        else:
            diff_penalty = np.zeros(3)[:, None]

        diff_object = diff_cov + diff_penalty

        normal     = diff_mean / np.linalg.norm(diff_mean, ord=2)
        projection = np.eye(self.X.shape[1]) - normal.dot(normal.T)

        S          = np.dot(projection, diff_object)
        direction  = S / np.linalg.norm(S)

        return normal, direction