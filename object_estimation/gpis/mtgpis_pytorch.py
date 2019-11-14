#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import torch
# import numpy as np

# class MultiTaskGaussianProcessImplicitSurfaces:
#     def __init__(self, X, Y, T, kernel, task_kernel_params, sigma=0.2, m=1, c=100, z_limit=0.03):

#         self.X  = torch.cat(X)
#         self.Y  = torch.cat(Y)
#         self.T  = torch.cat([torch.ones(X[i].shape[0],1, dtype=torch.long)*T[i] for i in range(len(X))])

#         self.sigma_y = torch.Tensor([sigma])
#         self.m       = m
#         self.c       = c
#         self.z_limit = z_limit

#         self.kernel  = kernel
#         self.task_kernel_params = task_kernel_params

#         self.KX = self.kernel(self.X, self.X)
#         self.Y_m = self.Y - self.m


#     def task_params_to_psd(self , task_kernel_params):
#         # task_psd = torch.exp(torch.triu(task_kernel_params))
#         # return torch.mm(task_psd.T, task_psd)

#         task_psd = torch.triu(torch.exp(task_kernel_params))
#         return torch.mm(task_psd.T, task_psd)


#     def predict(self, x, t):
#         t   = torch.ones(x.size()[0], dtype=torch.long) * t

#         Kt  = self.task_kernel[t, t][:, None]
#         KTt = self.task_kernel[self.T, t.T]

#         Kx  = ( torch.ones(len(x)) * (1.0 / float(self.kernel.params[0])) )[:, None]
#         KXx = self.kernel(self.X, x)

#         kk  = Kt  * Kx
#         k   = KTt * KXx

#         # sum ( A * B.T, axis=0 ) = diag( A.dot(B) )
#         invL_k = self.invL.mm(k)

#         mean = self.m + k.T.mm(self.invK).mm(self.Y_m)
#         cov  = kk - torch.sum(invL_k * invL_k, dim=0)[:, None] + self.sigma_y**2
#         # import ipdb; ipdb.set_trace()

#         return mean, cov

#     def compute_grad(self, flag):
#         # self.learning_params.requires_grad = flag
#         self.task_kernel_params.requires_grad = flag
#         # self.kernel.params.requires_grad = flag
#         # self.sigma_y.requires_grad = flag

#     def log_likehood(self):
#         # task_kernel_params = torch.zeros(2,2)
#         # task_kernel_params[0][0] = self.learning_params[0]
#         # task_kernel_params[0][1] = self.learning_params[1]
#         # task_kernel_params[1][1] = self.learning_params[2]

#         task_kernel = self.task_params_to_psd(self.task_kernel_params)
#         # task_kernel /= task_kernel[1][1]
#         # print task_kernel
#         KT = task_kernel[self.T, self.T.T]
#         # KX = self.kernel(self.X, self.X)
#         K = KT * self.KX + torch.eye(self.X.size()[0]) * (self.sigma_y**2 + 10e-6)

#         invKY = torch.solve(self.Y_m, K)[0]
#         # import ipdb; ipdb.set_trace()

#         return  torch.logdet(K) + self.Y_m.T.mm(invKY)

#     def learning(self, max_iter=40):

#         self.compute_grad(True)
#         params = [self.task_kernel_params]
#         # params = [self.task_kernel_params] + [self.kernel.params] + [self.sigma_y]
#         optimizer = torch.optim.Adam(params, lr=0.1)
#         # optimizer = torch.optim.LBFGS(params, history_size=20, max_iter=100)
#         # optimizer = torch.optim.SGD(params, lr=1e-3)

#         print "params:", params
#         for i in range(max_iter):
#             optimizer.zero_grad()
#             f = self.log_likehood()
#             f.backward()

#             if i % 20 == 0:
#                 print "-------------------------------------"
#                 # print self.learning_params.grad
#                 print self.task_kernel_params.grad
#                 print f.item()
#             # def closure():
#             #     optimizer.zero_grad()
#             #     with torch.autograd.detect_anomaly():
#             #         f = self.log_likehood()
#             #         f.backward()
#             #     return f
#             # optimizer.step(closure)
#             optimizer.step()

#         self.compute_grad(False)


#         # self.task_kernel_params[0][0] = self.learning_params[0]
#         # self.task_kernel_params[0][1] = self.learning_params[1]
#         # self.task_kernel_params[1][1] = self.learning_params[2]

#         self.task_kernel = self.task_params_to_psd(self.task_kernel_params)
#         # self.task_kernel /= self.task_kernel[1][1]
#         print self.task_kernel_params
#         print "task:", self.task_kernel
#         print "sigma:", self.sigma_y
#         print "kernel:", self.kernel.params

#         KT = self.task_kernel[self.T, self.T.T]
#         L  = torch.cholesky(KT * self.KX + torch.eye(self.X.size()[0]) * (self.sigma_y**2 + 10e-6))

#         self.invL = torch.solve(torch.eye(L.size()[0]), L)[0]
#         self.invK = torch.mm(self.invL.T, self.invL)

#     def predict_direction(self, x, t):
#         t   = torch.ones(x.size()[0], dtype=torch.long) * t
#         KTt = self.task_kernel[self.T, t.T]

#         diff_Kx = self.kernel.derivative(x, self.X)
#         KXx     = self.kernel(self.X, x)

#         kk      = KTt * diff_Kx
#         k       = KTt * KXx

#         diff_mean  = kk.T.mm(self.invK).mm(self.Y - self.m)
#         diff_cov   = -2 * (k.T.mm(self.invK).mm(kk)).T

#         if x[0][2] < self.z_limit:
#             diff_penalty = torch.Tensor([0, 0, -2 * self.c * (x[0][2] - self.z_limit)])[:, None]
#         else:
#             diff_penalty = torch.zeros(3)[:, None]

#         diff_object = diff_cov + diff_penalty

#         normal     = diff_mean / torch.norm(diff_mean, dim=0)

#         projection = torch.eye(self.X.size()[1]) - normal.mm(normal.T)
#         S          = torch.mm(projection, diff_object)
#         direction  = S / torch.norm(S)

#         return normal, direction




                # try:
        #     invK = torch.inverse(K)
        # except:
        #     print 'hogehoge'
        #     X = self.X
        #     sigma = torch.exp(self.sigma_y)
        #     import ipdb; ipdb.set_trace()
        # invL = torch.solve(torch.eye(L.size()[0]), L)[0]
        # invK = torch.mm(invL.T, invL)



        #!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

class MultiTaskGaussianProcessImplicitSurfaces:
    def __init__(self, X, Y, T, kernel, task_kernel_params, sigma=0.2, m=1, c=100, z_limit=0.03):

        self.X  = torch.cat(X)
        self.Y  = torch.cat(Y)
        self.T  = torch.cat([torch.ones(X[i].shape[0],1, dtype=torch.long)*T[i] for i in range(len(X))])

        self.sigma_y = torch.Tensor([sigma])
        self.m       = m
        self.c       = c
        self.z_limit = z_limit

        self.kernel  = kernel
        self.task_kernel_params = torch.triu(task_kernel_params)
        print self.task_kernel_params

        self.learning_params = torch.Tensor([self.task_kernel_params[0][0], self.task_kernel_params[0][1],
                                             self.task_kernel_params[1][1]])
        self.KX = self.kernel(self.X, self.X)
        self.Y_m = self.Y - self.m


    def task_params_to_psd(self , task_kernel_params):
        # task_psd = torch.exp(torch.triu(task_kernel_params))
        # return torch.mm(task_psd.T, task_psd)

        task_psd = torch.exp(task_kernel_params)
        return torch.mm(task_psd.T, task_psd)


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
        cov  = kk - torch.sum(invL_k * invL_k, dim=0)[:, None] + self.sigma_y**2
        # import ipdb; ipdb.set_trace()

        return mean, cov

    def compute_grad(self, flag):
        self.learning_params.requires_grad = flag
        # self.task_kernel_params.requires_grad = flag
        # self.kernel.params.requires_grad = flag
        # self.sigma_y.requires_grad = flag

    def log_likehood(self):
        task_kernel_params = torch.zeros(2,2)
        task_kernel_params[0][0] = self.learning_params[0]
        task_kernel_params[0][1] = self.learning_params[1]
        task_kernel_params[1][1] = self.learning_params[2]

        task_kernel = self.task_params_to_psd(task_kernel_params)
        # print task_kernel
        KT = task_kernel[self.T, self.T.T]
        # KX = self.kernel(self.X, self.X)
        K = KT * self.KX + torch.eye(self.X.size()[0]) * (self.sigma_y**2 + 10e-6)

        invKY = torch.solve(self.Y_m, K)[0]
        # import ipdb; ipdb.set_trace()

        return  torch.logdet(K) + self.Y_m.T.mm(invKY)

    def learning(self, max_iter=40):

        self.compute_grad(True)
        params = [self.learning_params]
        # params = [self.task_kernel_params] + [self.kernel.params] + [self.sigma_y]
        optimizer = torch.optim.Adam(params, lr=0.1)
        # optimizer = torch.optim.LBFGS(params, history_size=20, max_iter=100)
        # optimizer = torch.optim.SGD(params, lr=1e-3)

        print "params:", params
        for i in range(max_iter):
            optimizer.zero_grad()
            f = self.log_likehood()
            f.backward()

            if i % 20 == 0:
                print "-------------------------------------"
                print self.learning_params.grad
                # print self.task_kernel_params.grad
                print f.item()
            # def closure():
            #     optimizer.zero_grad()
            #     with torch.autograd.detect_anomaly():
            #         f = self.log_likehood()
            #         f.backward()
            #     return f
            # optimizer.step(closure)
            optimizer.step()

        self.compute_grad(False)


        self.task_kernel_params[0][0] = self.learning_params[0]
        self.task_kernel_params[0][1] = self.learning_params[1]
        self.task_kernel_params[1][1] = self.learning_params[2]

        self.task_kernel = self.task_params_to_psd(self.task_kernel_params)
        # self.task_kernel /= self.task_kernel[1][1]
        print self.task_kernel_params
        print "task:", self.task_kernel
        print "sigma:", self.sigma_y
        print "kernel:", self.kernel.params

        KT = self.task_kernel[self.T, self.T.T]
        L  = torch.cholesky(KT * self.KX + torch.eye(self.X.size()[0]) * (self.sigma_y**2 + 10e-6))

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



