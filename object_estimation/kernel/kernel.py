#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch



class ExponentialKernel:
    def __init__(self, params):
        self.params = params

    def __call__(self, x, y):
        sigma_x = np.zeros((len(self.params), len(self.params)))
        for i in range(len(self.params)):
            sigma_x[i][i] = self.params[i] ** 2

        return np.exp( (x-y).T.dot(sigma_x).dot(x-y) )

    def derivative(self, x, y):
        return -(x - y) / self.params[0]**2 * (np.exp((x - y).T.dot(x - y) / (2 * self.params[0]**2)))



class InverseMultiquadricKernel:
    def __init__(self, params):
        self.params = params

    def __call__(self, x, y):

        # sum_x_2 = torch.sum(torch.from_numpy(x).float()**2, dim=1)[:, None]
        # sum_y_2 = torch.sum(torch.from_numpy(y).float()**2, dim=1)[:, None]

        sum_x_2 = np.sum(x**2, axis=1).reshape(-1,1)
        sum_y_2 = np.sum(y**2, axis=1).reshape(-1,1)

        return  pow(sum_x_2 - 2*np.dot(x, y.T) + sum_y_2.T + self.params[0]**2, -0.5)

    # def derivative(self, x, y):
    #     return -(x - y) * pow(np.linalg.norm(x - y, ord=2)**2 + self.params[0]**2, -3/2)

    def derivative(self, x, y):

        # xx = torch.from_numpy(x).float()
        # yy = torch.from_numpy(y).float()
        # res = torch.zeros(xx.size()[0], yy.size()[0], xx.size()[1])

        # sum_x_2 = torch.sum(xx**2, dim=1)[:, None]
        # sum_y_2 = torch.sum(yy**2, dim=1)[:, None]
        # tmp = pow(sum_x_2 - 2*torch.mm(xx, yy.T) + sum_y_2.T + self.params[0]**2, -1.5)
        # for i in range(xx.size()[1]):
        #     x_dim = xx[:, i][:, None]
        #     y_dim = yy[:, i]

        #     res[:,:,i] =  -(x_dim - y_dim) * tmp

        res = np.zeros((x.shape[0], y.shape[0], x.shape[1]))
        sum_x_2 = np.sum(x**2, axis=1).reshape(-1,1)
        sum_y_2 = np.sum(y**2, axis=1).reshape(-1,1)
        tmp = pow(sum_x_2 - 2*np.dot(x, y.T) + sum_y_2.T + self.params[0]**2, -1.5)

        for i in range(x.shape[1]):
            x_dim = x[:, i][:, None]
            y_dim = y[:, i]

            res[:,:,i] =  -(x_dim - y_dim) * tmp

        return res[0]


class InverseMultiquadricKernelPytouch:
    def __init__(self, params):
        self.params = params

    def __call__(self, x, y):
        sum_x_2 = torch.sum(x**2, dim=1)[:, None]
        sum_y_2 = torch.sum(y**2, dim=1)[:, None]

        return  pow(sum_x_2 - 2 * torch.mm(x, y.T) + sum_y_2.T + self.params[0]**2, -0.5)

    def derivative(self, x, y):
        res = torch.zeros(x.size()[0], y.size()[0], x.size()[1])

        sum_x_2 = torch.sum(x**2, dim=1)[:, None]
        sum_y_2 = torch.sum(y**2, dim=1)[:, None]
        tmp = pow(sum_x_2 - 2 * torch.mm(x, y.T) + sum_y_2.T + self.params[0]**2, -1.5)

        for i in range(x.size()[1]):
            x_dim = x[:, i][:, None]
            y_dim = y[:, i]

            res[:,:,i] =  -(x_dim - y_dim) * tmp

        return res[0]


# class ExponentialKernel_params_2:
#     def __init__(self, params):
#         self.params = params

#     def __call__(self, x, y):
#         return self.params[0]**2 * np.exp(-(x - y)**2 / (2 * self.params[1]**2))

#     def derivative(self, x, y):
#         return -(x - y) / self.params[1]**2 * (self.params[0]**2 * np.exp(-(x - y)**2 / (2 * self.params[1]**2)))

# class ExponentialKernel_params_1:
#     def __init__(self, params):
#         self.params = params

#     def __call__(self, x, y):
#         return np.exp(-(x - y)**2 / (2 * self.params[0]**2))

#     def derivative(self, x, y):
#         return -(x - y) / self.params[0]**2 * np.exp(-(x - y)**2 / (2 * self.params[0]**2))


# class ExponentialKernel_params_3:
#     def __init__(self, params):
#         self.params = params

#     def __call__(self, x, y):
#         return np.exp(-(x - y)**2 / (2 * self.params[0]**2))

#     def derivative(self, x, y):
#         return -(x - y) / self.params[0]**2 * np.exp(-(x - y)**2 / (2 * self.params[0]**2))
