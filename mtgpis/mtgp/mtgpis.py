#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

class MultiTaskGaussianProcessImplicitSurfaces:
    def __init__(self, X, Y, T, kernel, task_kernel_params=None, sigma=0.1):

        self.X  = torch.cat(X)
        self.Y  = torch.cat(Y)
        self.T  = torch.cat([torch.ones(X[i].shape[0],1, dtype=torch.long)*T[i] for i in range(len(X))])

        self.m       = 1
        # self.sigma   = torch.tensor(np.log(sigma))
        self.sigma = sigma
        self.kernel  = kernel
        self.N_task  = self.T.unique().shape[0]
        # self.task_kernel_params = torch.rand(self.N_task, self.N_task)

        # if task_kernel_params is None:
        #     self.task_kernel_params = torch.tensor([[10e-4,10e-4],[10e-4,10e-4]])
        # else:
        self.task_kernel_params = task_kernel_params

        self.task_kernel = self.task_params_to_psd()

        KX = self.kernel(self.X, self.X)
        KT = self.task_kernel[self.T, self.T.T]
        L  = torch.cholesky(KT * KX + torch.eye(self.X.size()[0]) * torch.exp(self.sigma))

        self.Y_m = self.Y - self.m
        self.invL = torch.solve(torch.eye(L.size()[0]), L)[0]
        self.invK = torch.mm(self.invL.T, self.invL)

    def task_params_to_psd(self):
        u = torch.triu(torch.exp(self.task_kernel_params))
        return u.t().mm(u)

    def predict(self, x, t):
        t   = torch.ones(x.size()[0], dtype=torch.long) * t

        Kt  = self.task_kernel[t, t][:, None]
        KTt = self.task_kernel[self.T, t.T]

        Kx  = ( torch.ones(len(x)) * (1.0 / float(self.kernel.params[0])) )[:, None]
        KXx = self.kernel(self.X, x)

        kk  = Kt  * Kx
        k   = KTt * KXx

        # sum ( A * B.T, axis=0 ) = diag( A.dot(B) )
        invL_k = self.invL.mm(k)

        mean = self.m + k.T.mm(self.invK).mm(self.Y_m)
        cov  = kk - torch.sum(invL_k * invL_k, dim=0)[:, None]
        # import ipdb; ipdb.set_trace()

        return mean, cov

    def compute_grad(self, flag):
        self.task_kernel_params.requires_grad = flag
        # self.kernel.params.requires_grad = flag
        # self.sigma.requires_grad = flag

    def log_likehood(self):
        task_kernel = self.task_params_to_psd()

        K     = task_kernel[self.T, self.T.T] * self.kernel(self.X, self.X) \
                      + torch.eye(self.X.size()[0]) * torch.exp(self.sigma)
        invKY = torch.solve(self.Y_m, K)[0]

        return  torch.logdet(K) + self.Y_m.T.mm(invKY)

    def learning(self, max_iter=5000, lr=0.0001):

        self.compute_grad(True)
        params = [self.task_kernel_params] + [self.kernel.params] + [self.sigma]
        optimizer = torch.optim.Adam(params, lr=lr)
        # optimizer = torch.optim.LBFGS(params, history_size=20, max_iter=20,lr=0.1)
        # optimizer = torch.optim.SGD(params, lr=1e-3)

        # print("params:", params)
        for i in range(max_iter):
            optimizer.zero_grad()
            f = self.log_likehood()
            f.backward()

            # if i % 20 == 0:
            #     print("-------------------------------------")
            #     # print self.task_kernel_params.grad
            #     print(f.item())
            # def closure():
            #     optimizer.zero_grad()
            #     with torch.autograd.detect_anomaly():
            #         f = self.log_likehood()
            #         f.backward()
            #     return f
            # optimizer.step(closure)
            optimizer.step()
        self.compute_grad(False)

        self.task_kernel = self.task_params_to_psd()
        print("task:", self.task_kernel)
        print("sigma:", self.sigma)
        print("kernel:", self.kernel.params)
        print(self.task_kernel_params)

        L  = torch.cholesky(self.task_kernel[self.T, self.T.T] * self.kernel(self.X, self.X)\
                                 + torch.eye(self.X.size()[0]) * torch.exp(self.sigma))

        self.invL = torch.solve(torch.eye(L.size()[0]), L)[0]
        self.invK = torch.mm(self.invL.T, self.invL)

    def predict_direction(self, x, t):
        t   = torch.ones(x.size()[0], dtype=torch.long) * t
        KTt = self.task_kernel[self.T, t.T]

        diff_Kx = self.kernel.derivative(x, self.X)
        KXx     = self.kernel(self.X, x)

        kk      = KTt * diff_Kx
        k       = KTt * KXx

        diff_mean  = kk.T.mm(self.invK).mm(self.Y - self.m)
        diff_cov   = -2 * (k.T.mm(self.invK).mm(kk)).T

        normal     = diff_mean / torch.norm(diff_mean, dim=0)

        projection = torch.eye(self.X.size()[1]) - normal.mm(normal.T)
        S          = torch.mm(projection, diff_cov)
        direction  = S / torch.norm(S)

        return normal, direction