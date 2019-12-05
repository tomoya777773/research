#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np


class GaussianProcessImplicitSurfaces:
    def __init__(self, X, Y, kernel, sigma=0.1):

        self.X  = X
        self.Y  = Y

        self.sigma  = sigma
        self.m      = 1
        self.kernel = kernel

        self.KX = self.kernel(self.X, self.X)
        self.Y_m = self.Y - self.m

        L  = torch.cholesky(self.KX + torch.eye(self.X.size()[0]) * torch.exp(self.sigma))

        self.invL = torch.solve(torch.eye(L.size()[0]), L)[0]
        self.invK = torch.mm(self.invL.T, self.invL)

    def predict(self, x):
        Kx  = ( torch.ones(len(x)) * (1.0 / float(self.kernel.params[0])) )[:, None]
        KXx = self.kernel(self.X, x)

        # sum ( A * B.T, axis=0 ) = diag( A.dot(B) )
        invL_k = self.invL.mm(KXx)

        mean = self.m + KXx.T.mm(self.invK).mm(self.Y_m)
        cov  = Kx - torch.sum(invL_k * invL_k, dim=0)[:, None]

        return mean, cov

    def compute_grad(self, flag):
        self.kernel.params.requires_grad = flag
        self.sigma.requires_grad = flag

    def log_likehood(self):
        K = self.kernel(self.X, self.X) + torch.eye(self.X.size()[0]) * torch.exp(self.sigma)
        invKY = torch.solve(self.Y_m, K)[0]
        return  torch.logdet(K) + self.Y_m.T.mm(invKY)

    def learning(self, max_iter=5000, lr=0.0001):

        self.compute_grad(True)
        # params = [self.task_kernel_params]
        # params = [self.learning_params]
        params = [self.kernel.params] + [self.sigma]
        optimizer = torch.optim.Adam(params, lr=lr)
        # optimizer = torch.optim.LBFGS(params, history_size=20, max_iter=100)
        # optimizer = torch.optim.SGD(params, lr=1e-3)

        print("params:", params)
        for i in range(max_iter):
            optimizer.zero_grad()
            f = self.log_likehood()

            f.backward()

            if i % 20 == 0:
                print("-------------------------------------")
                # print self.learning_params.grad
                print(f.item())
            # def closure():
            #     optimizer.zero_grad()
            #     with torch.autograd.detect_anomaly():
            #         f = self.log_likehood()
            #         f.backward()
            #     return f
            # optimizer.step(closure)
            optimizer.step()

        self.compute_grad(False)



        print("sigma:", self.sigma)
        print("kernel:", self.kernel.params)

        self.KX = self.kernel(self.X, self.X)
        L  = torch.cholesky(self.KX + torch.eye(self.X.size()[0]) * torch.exp(self.sigma))

        self.invL = torch.solve(torch.eye(L.size()[0]), L)[0]
        self.invK = torch.mm(self.invL.T, self.invL)

    def predict_direction(self, x):
        diff_Kx = self.kernel.derivative(x, self.X)
        KXx     = self.kernel(self.X, x)

        diff_mean = diff_Kx.T.mm(self.invK).mm(self.Y_m)
        diff_cov  = -2 * (KXx.T.mm(self.invK).mm(diff_Kx)).T

        normal     = diff_mean / torch.norm(diff_mean, dim=0)

        projection = torch.eye(self.X.size()[1]) - normal.mm(normal.T)
        S          = torch.mm(projection, diff_cov)
        direction  = S / torch.norm(S)

        return normal, direction




                # try:
        #     invK = torch.inverse(K)
        # except:
        #     print 'hogehoge'
        #     X = self.X
        #     sigma = torch.exp(self.sigma)
        #     import ipdb; ipdb.set_trace()
        # invL = torch.solve(torch.eye(L.size()[0]), L)[0]
        # invK = torch.mm(invL.T, invL)