#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from scipy.optimize import fmin, fmin_bfgs, fmin_cg, fmin_powell, fmin_l_bfgs_b
import optuna

class MultiTaskGaussianProcessImplicitSurfaces:
    def __init__(self, X, Y, T, kernel):
        self.X1 = X[0]
        self.X2 = X[1]
        self.Y1 = Y[0]
        self.Y2 = Y[1]
        self.X  = np.concatenate(X)
        self.Y  = np.concatenate(Y)
        self.T  = np.concatenate([np.ones(X[i][:, 0].shape, dtype=np.long)*T[i] for i in range(len(X))])[:, None]

        self.kernel = kernel

        N_task = np.unique(self.T).shape[0]
        self.task_kernel_params = np.triu(np.random.rand(N_task, N_task))

        self.sigma_y = np.array([0.3])
        self.m       = 1
        self.c       = 100
        self.z_limit = 0.02

    def task_to_psd_matrix(self, task_kernel_params):
        L = np.exp(task_kernel_params)
        return np.dot(L.T, L)

    def calculate_inv(self, K):
        L = np.linalg.cholesky(K)
        inv_L = np.linalg.solve(L, np.eye(L.shape[0]))
        inv_K = np.dot(inv_L.T, inv_L)
        return inv_K

    def predict(self, x, t):
        t           = np.reshape(np.ones(x.shape[0], dtype=np.long) * t, (-1, 1))
        task_kernel = self.task_to_psd_matrix(self.task_kernel_params)

        Kt  = task_kernel[t, t]
        KTt = task_kernel[self.T, t.T]
        KT  = task_kernel[self.T, self.T.T]

        Kx  = np.diag(self.kernel(x, x))[:, None]
        KXx = self.kernel(self.X, x)
        KX  = self.kernel(self.X, self.X)

        kk  = Kt  * Kx
        k   = KTt * KXx
        Eye = np.eye(self.X.shape[0]) * self.sigma_y**2
        K   = KT  * KX + Eye

        invK = self.calculate_inv(K)

        mean = self.m + k.T.dot(invK).dot(self.Y - self.m)
        cov  = kk - np.reshape(np.diag(k.T.dot(invK).dot(k)), (-1,1)) + self.sigma_y**2

        return mean, cov

    def log_likelihood(self, trial):

        l1 = trial.suggest_uniform('l1', 0.0001, 100)
        l2 = trial.suggest_uniform('l2', 0.0001, 100)
        l3 = trial.suggest_uniform('l3', 0.0001, 100)

        L = np.exp(np.array( [ [ l1, l2 ], [ 0, l3 ] ] ))
        task_kernel        = np.dot(L.T, L)

        KT = task_kernel[self.T, self.T.T]
        KX = self.kernel(self.X, self.X)

        Eye = np.eye(self.X.shape[0]) * self.sigma_y**2
        K   = KT * KX + Eye

        return  np.linalg.slogdet(K)[1] + (self.Y - self.m).T.dot(self.calculate_inv(K)).dot(self.Y - self.m)


    def gradient(self, params):
        # print params
        # if type(params[0])==np.ndarray:
        #     print params
        #     params = params[0]
        # if params.shape[0] == 1:
        #     params = params[0]
        #     print params,type(params)
        # if len(params) == 1:
        #     params = params[0]
        task_kernel_params = np.array([[params[0], params[1]], [0, params[2]]])
        task_kernel = self.task_to_psd_matrix(task_kernel_params)

        KT = task_kernel[self.T, self.T.T]
        KX = self.kernel(self.X, self.X)

        Eye = np.eye(self.X.shape[0]) * self.sigma_y**2
        K   = KT * KX + Eye
        invK = self.calculate_inv(K)
        invKY = invK.dot(self.Y)

        grad_l0 = np.array([ [ 2*np.exp(2*params[0])        , np.exp(params[0] + params[1]) ],\
                             [ np.exp(params[0] + params[1]),                            0  ]])
        grad_l1 = np.array([ [ 0                            , np.exp(params[0] + params[1]) ],\
                             [ np.exp(params[0] + params[1]), 2*np.exp(2*params[1])         ]])
        grad_l2 = np.array([ [ 0 ,                     0 ],\
                             [ 0 , 2*np.exp(2*params[2]) ]])

        N = len(params)
        grad_l = [grad_l0, grad_l1, grad_l2]
        grad = np.zeros(N)

        # print grad_l
        # print np.trace(invK.dot(grad_l[0]))
        for i in range(N):
            grad_KT = grad_l[i][self.T, self.T.T]

            grad[i] = np.trace(invK.dot(grad_KT)) - invKY.T.dot(grad_KT).dot(invKY)
        # print "grad:", grad
        return grad



    def learn_params(self):
        study = optuna.create_study()
        study.optimize(self.log_likelihood, n_trials=30)

        print(study.best_params)
        res = study.best_params
        print(res["l1"])
        print(res)
        self.task_kernel_params = np.array([[res["l1"], res["l2"]], [0, res["l3"]]])

        # print("optimoize param:", xopt)
        print("task_kernel_params:", self.task_kernel_params)
        print("task_kernel:", self.task_to_psd_matrix(self.task_kernel_params))


    def predict_direction(self, x, t):
        # self.task_kernel_params[0][1] = 0.0001
        # self.task_kernel_params[1][1] = 0.0001

        t = np.reshape(np.ones(x.shape[0], dtype=np.long) * t, (-1, 1))
        task_kernel = self.task_to_psd_matrix(self.task_kernel_params)

        Kt  = task_kernel[t, t]
        KTt = task_kernel[self.T, t.T]
        KT  = task_kernel[self.T, self.T.T]

        diff_Kx = self.kernel.derivative(x, self.X)[0]
        KXx     = self.kernel(self.X, x)
        KX      = self.kernel(self.X, self.X)

        kk   = KTt * diff_Kx
        k    = KTt * KXx
        K    = KT  * KX + np.eye(self.X.shape[0]) * self.sigma_y**2
        invK = self.calculate_inv(K)


        diff_mean  = kk.T.dot(invK).dot(self.Y - self.m)
        diff_cov   = -2 * (k.T.dot(invK).dot(kk)).T

        if x[0][2] < self.z_limit:
            diff_penalty = np.array([0, 0, -2 * self.c * (x[0][2] - self.z_limit)])[:, None]
        else:
            diff_penalty = np.zeros(3)[:, None]

        diff_object = diff_cov + diff_penalty

        normal     = diff_mean / np.linalg.norm(diff_mean, ord=2)
        projection = np.eye(self.X.shape[1]) - normal.dot(normal.T)

        S         = np.dot(projection, diff_object)
        direction = S / np.linalg.norm(S)

        # print "================"
        # print "Kt:", Kt.shape
        # print "KTt:", KTt.shape
        # print "KT:", KT.shape

        # print "diff_Kx:", diff_Kx.shape
        # print "KXx:", KXx.shape
        # print "KX:", KX.shape

        # print diff_cov
        # print projection
        # print S
        # print direction
        # print "================"

        return normal, direction