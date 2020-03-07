# -*- coding: utf-8 -*-
import numpy as np
import math
import sympy
length=0.2
m=1
siguma=0.3
a=0.1

class Find_position:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.x = np.array([X[X.shape[0]-1]]).T
        self.zeros = np.zeros((X.shape[0],X.shape[0]))
        # print self.X.shape[1]
        # print self.X[self.X.shape[0]-1][0]
        # print self.X[0][0]

    def kernel_func(self, x1, x2): #kernel
        norumu2 = np.linalg.norm(x1-x2)
        kernel = pow(norumu2*norumu2 + length*length, -1/2)
        return kernel

    def diff_kernel_func(self, x1, x2):
        norumu2 = np.linalg.norm(x1-x2)
        diff_kernel = -norumu2 * pow(norumu2*norumu2 + length*length, -3/2)
        return diff_kernel

    def position(self):
        """カーネルとカーネルの微分の列ベクトル作成"""

        kernel_x = self.zeros[0]
        for i in range(self.zeros.shape[0]):
            kernel_x[i] = self.kernel_func(self.X[i], self.X[self.X.shape[0]-1])  #k(x)
        kernel_x = np.array([kernel_x]).T

        diff_kernel_x = np.zeros((self.zeros.shape[0],self.X.shape[1]))
        for i in range(self.zeros.shape[0]):
            for j in range(self.X.shape[1]):
                diff_kernel_x[i][j] = self.diff_kernel_func(self.X[self.X.shape[0]-1][j], self.X[i][j])


        """カーネル行列の作成"""
        Kernel_x = self.zeros
        for i in range(self.zeros.shape[0]):
            for j in range(self.zeros.shape[0]):
                Kernel_x[i][j] = self.kernel_func(self.X[i], self.X[j])  #K

        G = Kernel_x + siguma*siguma * np.identity(self.zeros.shape[0])
        invG = np.linalg.inv(G)
        b = np.dot(invG, self.Y -m)

        """平均と分散"""
        mean = m + np.dot(kernel_x.T, b)
        var = 1/length - np.dot(np.dot(kernel_x.T, invG), kernel_x)

        """平均と分散の微分"""
        diff_mean = (np.dot(b.T, diff_kernel_x)).T
        diff_var = (1/length - 2 * np.dot(np.dot(kernel_x.T, invG), diff_kernel_x)).T

        """法線"""
        normal = diff_mean / np.linalg.norm(diff_mean)

        """接平面"""
        projection_n = np.identity(self.X.shape[1]) - normal * normal.T

        """新しい方向"""
        S = np.dot(projection_n, diff_var)
        new_direction = a * S / np.linalg.norm(S)

        """新しい位置"""
        position = self.x + np.dot(projection_n, new_direction)

        pred_normal = np.zeros(2)
        pred_normal[0] = normal[0]
        pred_normal[1] = normal[1]
        pred_position = np.zeros(2)
        pred_position[0] = position[0][0]
        pred_position[1] = position[1][0]

        return pred_normal, pred_position

class Normal_position:
    def __init__(self, X, n):
        self.x = X[0]
        self.y = X[1]
        self.n = n
        self.a = -n[1]/n[0]
        self.c = self.y - self.a * self.x

    def solve_sim_equations(self):
        x1 = (-self.a*self.c + math.sqrt(self.a**2*self.c**2-(self.a+1)*(self.c**2-1))) / (self.a + 1)
        x2 = (-self.a*self.c - math.sqrt(self.a**2*self.c**2-(self.a+1)*(self.c**2-1))) / (self.a + 1)
        y1 = self.a*x1 + self.c
        y2 = self.a*x2 + self.c


        A = (self.x-x1)**2+(self.y-y1)**2
        B = (self.x-x2)**2+(self.y-y2)**2
        if A < B:
            new_x = x1
            new_y = y1
        elif A > B:
            new_x = x2
            new_y = y2

        new_position = np.zeros(2)
        new_position[0] = new_x
        new_position[1] = new_y
        # print "################"
        # print new_position
        # new_position = np.array((new_x, new_y))
        return new_position
        # a=np.array([new_position])
        # print type(new_position)

if __name__=='__main__':

    X = np.array([[1,0],[1,0.1]])
    Y = np.array([0, -1]).T
    normal = np.array([[1, 0]])

    fp = Find_position(X,Y)
    new_ = fp.position()
    new_normal = new_[0]
    pred_position = new_[1]
    # print new_normal
    # print pred_position

    nop = Normal_position(X[1], normal[0])

    normal_position = nop.solve_sim_equations()
    print "###################"
    # print normal_position

    X = np.append(X, [normal_position], axis=0)



    for i in range(3):
        print X[X.shape[0]-1][0]
        circle_detection = X[X.shape[0]-1][0]**2 + X[X.shape[0]-1][1]**2 - 1

        print circle_detection

        if circle_detection == 0:
            # X = np.append(X, new_position)
            Y = np.append(Y, 0)
            fp = Find_position(X,Y)
            new_ = fp.position()
            new_normal = new_[0]
            pred_position = new_[1]

            normal = np.append(normal, [new_normal], axis=0)
            X = np.append(X, [pred_position], axis=0)
            print normal
            print X


        elif circle_detection != 0:
            # print X[X.shape[0]-1]
            # print normal[normal.shape[0]-1]
            nop = Normal_position(X[X.shape[0]-1], normal[normal.shape[0]-1])
            normal_position = nop.solve_sim_equations()
            print normal_position
            X = np.append(X, [normal_position], axis=0)
            print X
