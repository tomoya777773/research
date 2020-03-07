# -*- coding: utf-8 -*-
import numpy as np
import math
import sympy
import matplotlib.pyplot as plt

length=0.2
m=1
siguma=0.3
a=0.03

class Find_position:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.x = np.array([X[X.shape[0]-1]]).T
        self.zeros = np.zeros((X.shape[0],X.shape[0]))
        # print(self.Y)

    def kernel_func(self, x1, x2): #kernel
        norumu2 = np.linalg.norm(x1-x2)
        kernel = pow(norumu2*norumu2 + length*length, -1/2)
        return kernel

    def diff_kernel_func(self, x1, x2, x3, x4):
        norumu2 = np.linalg.norm(x1-x2)
        diff_kernel = -(x3 - x4) * pow(norumu2*norumu2 + length*length, -3/2)
        return diff_kernel

    def normal_func(self):

        """カーネルとカーネルの微分の列ベクトル作成"""
        kernel_x = np.zeros((X.shape[0]))
        for i in range(self.zeros.shape[0]):
            kernel_x[i] = self.kernel_func(self.X[i], self.X[self.X.shape[0]-1])  #k(x)
        kernel_x = np.array([kernel_x]).T
        print "kernel:", kernel_x

        # diff_kernel_x = np.zeros((X.shape[0]))
        diff_kernel_x = np.zeros((self.zeros.shape[0],self.X.shape[1]))
        for i in range(self.zeros.shape[0]):
            for j in range(self.X.shape[1]):
                diff_kernel_x[i][j] = self.diff_kernel_func(self.X[self.X.shape[0]-1], self.X[i], self.X[self.X.shape[0]-1][j], self.X[i][j])

        # print("diff_kernel:", diff_kernel_x)

        """カーネル行列の作成"""
        Kernel_x = np.zeros((X.shape[0],X.shape[0]))
        for i in range(self.zeros.shape[0]):
            for j in range(self.zeros.shape[0]):
                Kernel_x[i][j] = self.kernel_func(self.X[i], self.X[j])  #K

        print Kernel_x
        G = Kernel_x + siguma*siguma * np.identity(self.zeros.shape[0])
        invG = np.linalg.inv(G)

        b = np.dot(invG, self.Y - m)
        # print("b:", b)

        """平均と分散"""
        mean = m + np.dot(kernel_x.T, b)
        var = 1/length - np.dot(np.dot(kernel_x.T, invG), kernel_x)

        """平均と分散の微分"""
        self.diff_mean = (np.dot(b.T, diff_kernel_x)).T
        self.diff_var = (- 2 * np.dot(np.dot(kernel_x.T, invG), diff_kernel_x)).T
        # print("diff_mean", self.diff_mean)

        """法線"""
        self.normal = self.diff_mean / np.linalg.norm(self.diff_mean)

        return self.normal

    def position_func(self):
        normal = self.normal_func()
        # print("normal:",normal)

        """接平面"""
        projection_n = np.identity(self.X.shape[1]) - np.dot(normal, normal.T)
        # print(projection_n)
        # print('###')
        """新しい方向"""
        S = np.dot(projection_n, self.diff_var)
        # print(S)
        new_direction = a * S / np.linalg.norm(S)

        """新しい位置"""
        position = self.x + np.dot(projection_n, new_direction)

        pred_position = np.zeros(2)
        pred_position[0] = position[0][0]
        pred_position[1] = position[1][0]

        return pred_position

class Normal_position:
    def __init__(self, X, a):
        self.x = X[0]
        self.y = X[1]
        self.a = a[1] / a[0]
        self.c = self.y - self.a * self.x
        # print("a:",self.a)
        print(self.c)


    def solve_sim_equations(self):
        x1 = (-self.a*self.c + math.sqrt(self.a**2 * self.c**2-(self.a**2+1)*(self.c**2-1))) / (self.a**2 + 1)
        x2 = (-self.a*self.c - math.sqrt(self.a**2 * self.c**2-(self.a**2+1)*(self.c**2-1))) / (self.a**2 + 1)
        y1 = self.a*x1 + self.c
        y2 = self.a*x2 + self.c
        print(x1,y1)
        print(x2,y2)

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

        return new_position

if __name__=='__main__':
    print('##################')
    X = np.array([[0.9,math.sqrt(0.19)],[0.8999, 0.4356]])
    # X = np.array([[0.1,math.sqrt(0.99)],[0.11, 0.99]])

    Y = np.array([0])
    normal = np.array([[1, 0]])


    count=0
    for i in range(50):
        count += 1
        print("count:", count)

        circle_detection = X[X.shape[0]-1][0]**2 + X[X.shape[0]-1][1]**2 - 1

        # print ("circle:", circle_detection)

        if round(circle_detection, 10) == 0:
            Y = np.append(Y, [0])
            Y = Y[:, np.newaxis]

            fp = Find_position(X, Y)
            pred_position = fp.position_func()

            # normal = np.append(normal, [new_normal], axis=0)
            X = np.append(X, [pred_position], axis=0)
            # print ("normal:",normal)
            # print ("X:", X)
            # print ("Y:", Y.T)
            # print ("normal:",normal)

            # print ("###################")


        elif round(circle_detection, 10) != 0:
            Y = np.append(Y, [-1])
            Y = Y[:, np.newaxis]

            fp = Find_position(X,Y)
            normal = fp.normal_func()

            nop = Normal_position(X[X.shape[0]-1], normal)
            normal_position = nop.solve_sim_equations()
            X = np.append(X, [normal_position], axis=0)

            # print ("X:", X)
            # print ("Y:", Y.T)
            # print ("normal:",normal)
            # print ("###################")
    print X
    print Y
    gp_data = np.array([[X.T[0]],X.T[1],[Y.T]])
    np.save('gp_data.npy', gp_data)
    print(X.T)
    plt.plot(X.T[0], X.T[1])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
