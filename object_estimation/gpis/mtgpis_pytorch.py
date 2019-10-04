#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

class MultiTaskGaussianProcessImplicitSurfaces:
    def __init__(self, X, Y, T, kernel, task_kernel_params, sigma=0.3, m=1, c=100, z_limit=0.03):
        self.X1 = X[0]
        self.X2 = X[1]
        self.Y1 = Y[0]
        self.Y2 = Y[1]
        self.X  = torch.cat(X)
        self.Y  = torch.cat(Y)
        self.T  = torch.cat([torch.ones(X[i].shape[0],1, dtype=torch.long)*T[i] for i in range(len(X))])

        self.sigma_y = sigma
        self.m       = m
        self.c       = c
        self.z_limit = z_limit
        self.kernel = kernel

        self.task_kernel_params = torch.Tensor(task_kernel_params)

        L   = torch.exp(torch.Tensor(self.task_kernel_params))
        self.task_kernel = torch.mm(L, L.T)

        self.KX   = self.kernel(self.X, self.X)
        KT        = self.task_kernel[self.T, self.T.T]
        L         = torch.cholesky(KT * self.KX + torch.eye(self.X.size()[0]) * self.sigma_y**2)
        self.invL = torch.solve(torch.eye(L.size()[0]), L)[0]
        self.invK = torch.mm(self.invL.T, self.invL)



    def task_to_psd_matrix(self, task_kernel_params):
        L   = torch.exp(torch.Tensor(task_kernel_params))
        res = torch.mm(L, L.T)
        # print res
        L   = np.exp(task_kernel_params)
        res = np.dot(L, L.T)
        # print res
        return res

    def calculate_inv(self, K):
        L = np.linalg.cholesky(K)
        inv_L = np.linalg.solve(L, np.eye(L.shape[0]))
        inv_K = np.dot(inv_L.T, inv_L)
        return inv_K

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

        mean = self.m + k.T.mm(self.invK).mm(self.Y - self.m)
        cov  = kk - torch.sum(invL_k * invL_k, dim=0)[:, None] + self.sigma_y**2

        return mean, cov

    def compute_grad(self, flag):
        self.task_kernel_params.requires_grad = flag

    def negative_log_likelihood(self):
        # task_kern = self.task_K()
        # # KT = [[task_kern[int(x)][int(y)] for x in self.T.flatten()] for y in self.T.flatten()]
        # KT = task_kern[self.T, self.T.t()]
        # K = KT*self.kern.K(self.X) + torch.eye(self.X.shape[0])*torch.exp(self.sigma)


        # invKY = torch.solve(self.Y, K)[0]
        # # logdet = torch.cholesky(K, upper=False).diag().log().sum()
        # logdet = torch.logdet(K)

        # return (logdet + self.Y.t().mm(invKY))


        # task_kernel_params = np.array( [ [ params[0], 0 ], [ params[1], params[2] ] ] )

        task_kernel_params = self.task_kernel_params

        L   = torch.exp(torch.Tensor(task_kernel_params))
        task_kernel = torch.mm(L, L.T)

        KT = task_kernel[self.T, self.T.T]
        # print KT
        # print torch.from_numpy(self.KX).float()
        # print KT * torch.from_numpy(self.KX).float()
        # print torch.eye(self.X.shape[0]) * self.sigma_y**2

        # K   = KT * torch.from_numpy(self.KX).float() + torch.eye(self.X.size()[0]) * self.sigma_y**2
        K   = KT * torch.from_numpy(self.KX).float() + torch.eye(self.X.shape[0]) * self.sigma_y**2

        L    = torch.cholesky(K)
        invL = torch.solve(torch.eye(L.size()[0]), L)[0]
        invK = torch.mm(invL.T, invL)

        Y_m = torch.from_numpy(self.Y - self.m).float()
        # print Y_m.T.size()
        # print invK.size()

        return  torch.logdet(K) + Y_m.T.mm(invK).mm(Y_m)




    def learning(self):
        max_iter = 100

        self.compute_grad(True)
        param = [self.task_kernel_params]
        # optimizer = torch.optim.Adam(param, lr=0.001)
        optimizer = torch.optim.LBFGS(param, history_size=20, max_iter=200)

        for i in range(max_iter):
            optimizer.zero_grad()
            f = self.negative_log_likelihood()
            f.backward()
            def closure():
                optimizer.zero_grad()
                with torch.autograd.detect_anomaly():
                    f = self.negative_log_likelihood()
                    f.backward()
                return f
            optimizer.step(closure)
        # optimizer.step()
        self.compute_grad(False)



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


    def predict_direction(self, x, t):
        t   = torch.ones(x.size()[0], dtype=torch.long) * t
        KTt = self.task_kernel[self.T, t.T]

        diff_Kx = self.kernel.derivative(x, self.X)
        KXx     = self.kernel(self.X, x)

        kk      = KTt * diff_Kx
        k       = KTt * KXx

        diff_mean  = kk.T.mm(self.invK).mm(self.Y - self.m)
        diff_cov   = -2 * (k.T.mm(self.invK).mm(kk)).T

        if x[0][2] < self.z_limit:
            diff_penalty = torch.Tensor([0, 0, -2 * self.c * (x[0][2] - self.z_limit)])[:, None]
        else:
            diff_penalty = torch.zeros(3)[:, None]

        diff_object = diff_cov + diff_penalty

        normal     = diff_mean / torch.norm(diff_mean, dim=0)

        projection = torch.eye(self.X.size()[1]) - normal.mm(normal.T)
        S          = torch.mm(projection, diff_object)
        direction  = S / torch.norm(S)

        return normal, direction