#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tqdm import tqdm
import time

import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import *
import scipy.optimize

import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Bool
from std_msgs.msg import Char

# from pyquaternion import Quaternion
import tf
from geometry_msgs.msg import Quaternion

length=0.2
m=1
siguma=0.3
# a=0.03
a=0.01
c = 100
z_limit = 1.02

class Path_Planning:

    """
    1　GPの平均と分散
    ２　凹凸判定
    ３　経路決定

    Attributes
    ----------
    X : float
        観測点(x,y,z)
    Y : -1, 0, 1
        観測点の接触を判断するラベル
    """

    def __init__(self, X, Y):

        """
        Parameters
        ----------
        X : float
            観測点(x,y,z)
        Y : -1, 0, 1
            観測点の接触を判断するラベル

        Notes
        ----------
        kernel_x : vector
            inverse-multiquadric kernel
        diff_kernel_x : matrix
            differential kernel_x
        Kernel_x : matrix
            kernel function
        """

        self.X = X
        self.Y = Y
        num = X.shape[0]

        """カーネル関数とカーネルの微分関数を定義"""
        kernel_func = lambda a,b : pow(np.linalg.norm(a-b)**2 + length**2 , -1/2)
        diff_kernel_func = lambda a,b,c,d : -(c-d) * pow(np.linalg.norm(a-b)**2 + length**2 , -3/2)

        """カーネルベクトル作成"""
        kernel_x = [kernel_func(self.X[i], self.X[num -1]) for i in range(num)]
        self.kernel_x = np.array(kernel_x)
        # print("kernel:", kernel_x)

        """カーネルの微分ベクトル作成"""
        self.diff_kernel_x = np.zeros((num, self.X.shape[1]))
        for i in range(num):
            for j in range(self.X.shape[1]):
                self.diff_kernel_x[i][j] = diff_kernel_func(self.X[num-1], self.X[i], self.X[num-1][j], self.X[i][j])
        self.diff_kernel_x = np.array(self.diff_kernel_x)
        # print("diff_kernel:", self.diff_kernel_x)

        """カーネル行列作成"""
        self.Kernel_x = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                self.Kernel_x[i][j] = kernel_func(self.X[i], self.X[j])  #K
        self.Kernel_x = np.array(self.Kernel_x)
        # print(self.Kernel_x)

        # """ヘッセ行列の項別の関数を定義"""
        # diff_xx = lambda a,b,c,d: 3*(c-d)**2 * pow(np.linalg.norm(a-b)**2 + length**2, -5/2) - pow(np.linalg.norm(a-b)**2 + length**2, -3/2)
        # diff_xy = lambda a,b,c,d: 3*(c[0]-d[0])*(c[1]-d[1])*pow(np.linalg.norm(a-b)**2 + length**2, -5/2)

        """G, bを求める"""
        G = self.Kernel_x + siguma**2 * np.identity(num)
        self.invG = np.linalg.inv(G)
        self.b = np.dot(self.invG, self.Y - m)
        # print("b:", self.b)

    """平均と分散"""
    def mean_and_var(self, current_position):

        """
        Parameters
        ----------
        X : float
            観測点
        current_position : float
            平均や分散を求めたい位置

        Returns
        ----------
        mean : float
            求めたい位置での平均
        var :float
            求めたい位置での分散
        """

        kernel_func = lambda a,b : pow(np.linalg.norm(a-b)**2 + length**2 , -1/2)
        kernel_x_ = [kernel_func(self.X[i], current_position) for i in range(self.X.shape[0])]
        self.kernel_x_ = np.array(kernel_x_)
        # print("self.kernel_x:", self.kernel_x_)

        mean = m + np.dot(self.kernel_x_, self.b)
        var = 1/length - np.dot(np.dot(self.kernel_x_,self. invG), self.kernel_x_)

        return mean, var

    def diff_mean_var(self, current_position):



    """凹凸判定"""
    def decision_func(self, x, y, z):

        """
        Parameters
        ----------
        x, y, z : float
            凹凸を判定する位置

        Returns
        ----------
        CC : 0 or 1 (0 : 凹, 1 : 凸)
            convexoconcave
        """

        position_list, mean_list, mean_zero_list, var, var_list = [],[],[],[],[]
        num = x.shape[0]
        for i in range(num):
            for j in range(num):
                for k in range(num):
                    po = [x[i],y[j],z[k]]
                    position_list.append(po)
                    mean_var = self.mean_and_var(po)
                    mean_list.append(abs(mean_var[0]))
                    var.append(mean_var[1])

                m_list = np.array(mean_list)
                n = m_list.argmin()

                mean_zero_list.append(position_list[n])
                var_list.append(var[n])

                position_list, mean_list, var = [],[],[]

        mean_zero_list = np.array(mean_zero_list)
        # var_list = np.reciprocal(var_list)

        mean_x = mean_zero_list[::, 0]
        mean_y = mean_zero_list[::, 1]
        mean_z = mean_zero_list[::, 2]

        regression_func = lambda param,x,y,z: z - (param[0]*x**2 + param[1]*y**2 + param[2]*x + param[3]*y + param[4])
        param = [0, 0, 0, 0, 0]
        optimized_param =  scipy.optimize.leastsq(regression_func, param, args=(x, y, z))[0]

        x = Symbol('x')
        y = Symbol('y')

        z = optimized_param[0]*x**2 + optimized_param[1]*y**2 + optimized_param[2]*x + optimized_param[3]*y + optimized_param[4]


        hesse00 = np.array(diff(diff(z, x), x), dtype=np.float64)
        hesse01 = np.array(diff(diff(z, y), x), dtype=np.float64)
        hesse10 = hesse01.copy()
        hesse11 = np.array(diff(diff(z, y), y), dtype=np.float64)

        Hesse_matrix = np.array([[hesse00,hesse01], [hesse10,hesse11]])

        la, v = np.linalg.eig(Hesse_matrix)
        print la

        cc = 0
        if la[0] > 0 and la[1] > 0: # 凸のとき
            cc = 1

        return cc

    """法線と方向"""
    def direction_func(self):

        """
        Parameters
        ----------

        Returns
        ----------
        normal : vector
            normal against object
        d : vector
            new direction on projection
        """

        """normal"""
        self.diff_mean = (np.dot(self.b.T, self.diff_kernel_x)).T
        normal = self.diff_mean / np.linalg.norm(self.diff_mean)
        # print("noemal:", normal)

        """projection"""
        projection_n = np.identity(self.X.shape[1]) - np.dot(normal, normal.T)

        """diff object function"""

        diff_var = (- 2 * np.dot(np.dot(self.kernel_x, self.invG), self.diff_kernel_x)).T
        diff_orbit =

        """新しい方向"""
        S = np.dot(projection_n, self.diff_var)
        new_direction = S / np.linalg.norm(S)
        d = np.dot(projection_n, new_direction)

        # print("b:", self.b)
        # print("diff_k:", self.diff_kernel_x)
        # print("diffffff:", self.diff_var)
        # print("normal:",normal)
        # print("p:", projection_n)
        # print("S:", S)
        # print("new:", new_direction)
        # print("d", d)
        return normal, d


class VREP_UR5:

    """
    Attributes
    ----------
    X : float
        観測点(x,y,z)
    Y : -1, 0, 1
        観測点の接触を判断するラベル
    """

    def __init__(self):

        """
        Publish
        ----------
        start_sim_pub : Vrep start
        stop_sim_pub : Vrep stop
        position_pub : send Vrep position
        orientation_pub : send vrep orientation

        Subscribe
        ----------
        position_sub : subscribe arm position
        force_sub : subscribe whether collisition

        Notes
        ----------
        position : Float32MultiArray
        orientation : Float32MultiArray
        """


        rospy.init_node('ur5_moving', disable_signals=True)

        self.start_sim_pub = rospy.Publisher('startSimulation', Bool, queue_size=10)
        self.stop_sim_pub = rospy.Publisher('stopSimulation', Bool, queue_size=10)

        self.position_pub = rospy.Publisher("IKTarget_position",  Float32MultiArray, queue_size=10)
        self.orientation_pub = rospy.Publisher("IKTarget_orientation", Float32MultiArray, queue_size=10)

        self.position_sub = rospy.Subscriber('arm_position', Float32MultiArray, self.callback_position)
        self.force_sub = rospy.Subscriber('collisition', Char, self.callback_force)

        self.init_rate = rospy.Rate(20)

        self.position = Float32MultiArray()
        self.orientation = Float32MultiArray()

        self.position.data = np.array([0.4, 0.01, 1.4])

        """データを読み込み"""
        self.X = np.load("spheroid_list.npy")
        self.Y = np.zeros((self.X.shape[0], 1))
        self.force = 0

        # print self.X.shape[0]
        # print self.Y

    def start_sim(self):
        print 'start simulation'
        self.sim_state = True
        frag = Bool()
        frag.data = True
        for _ in range(10):
            self.init_rate.sleep()
            self.start_sim_pub.publish(frag)

    def stop_sim(self):
        print 'stop simulation'
        frag = Bool()
        frag.data = True
        self.stop_sim_pub.publish(frag)

    def callback_force(self, msg):
        self.force = msg.data
        # print self.force

    def callback_position(self, msg):
        self.arm_position = msg.data

    def publisher_position(self, position):
        self.position_pub.publish(position)
        print "-----publish position-----"

    def publisher_orientation(self, orientation):
        self.orientation_pub.publish(orientation)
        print "-----publish orientation-----"

    def create_orientation(self, normal):

        # a =  math.atan(normal[1] / normal[0]) / np.pi * 180
        # if normal[2] < 0:
        #     a = a
        # elif normal[2] > 0:
        #     a = -a
        # q = Quaternion(axis = [normal[0], normal[1], normal[2]], angle=10)
        x = abs(normal[0])
        y = abs(normal[1])
        z = abs(normal[2])
        print "normal:",normal
        print "xyz:",x,y,z
        if normal[1] > 0 and normal[2] > 0:
            alpha = 90 + math.atan(z / y) / np.pi * 180
        elif normal[1] < 0 and normal[2] > 0:
            alpha = 270 - (math.atan(z / y) / np.pi * 180)
        elif normal[1] < 0 and normal[2] < 0:
            alpha = 270 + (math.atan(z / y) / np.pi * 180)
        elif normal[1] > 0 and normal[2] < 0:
            alpha = 90 - (math.atan(z / y) / np.pi * 180)

        if normal[0] > 0 and normal[2] > 0:
            beta = 270 - math.atan(z / x) / np.pi * 180
        elif normal[0] < 0 and normal[2] > 0:
            beta = 90 + (math.atan(z / x) / np.pi * 180)
        elif normal[0] < 0 and normal[2] < 0:
            beta = 90 - (math.atan(z / x) / np.pi * 180)
        elif normal[0] > 0 and normal[2] < 0:
            beta = 270 + (math.atan(z / x) / np.pi * 180)

        # alpha = 10
        # beta = 358

        # alpha = 180 - (math.atan(normal[1] / normal[2]) / np.pi * 180)
        # beta = - math.atan(normal[0] / normal[2]) / np.pi * 180
        # ganma = math.atan(normal[1] / normal[0]) / np.pi * 180
        ganma = 0
        orientation = np.array([alpha, beta, ganma])

        print "------eurler:", orientation

        orientation = np.array([alpha*np.pi/180, beta*np.pi/180, ganma])
        print "------eurler:", orientation
        # q = tf.transformations.quaternion_from_euler(orientation[0], orientation[1], orientation[2])
        # #
        # quaternion = [q[0], q[1], q[2], q[3]]
        # return quaternion

        return orientation


    def main(self):
        self.start_sim()
        # self.orientation.data = np.array([10, 358, 0])
        # self.publisher_orientation(self.orientation)

        print(self.position)

        """物体に当たるまで直進"""
        while True:
            if self.force == 0:
                print "-----not contact-----"

                self.position.data[2] -= 0.001
                self.publisher_position(self.position)

            elif self.force == 1:
                print "-----contact-----"

                self.X = np.vstack((self.X, self.arm_position))
                self.Y = np.vstack((self.Y, [0]))
                break
            rospy.sleep(0.1)


        self.X = np.reshape(self.X, (-1, 3))
        self.Y = np.reshape(self.Y, (-1, 1))

        # print self.X
        # print self.Y


        cnt = 1
        rospy.sleep(1)

        print "-----start serching-----"

        """
        after robot contacts object
            1. Judge unevenness(Concave or Convex)
            2. Perform each search
        """


        for i in tqdm(range(5)):
            w = 20
            current_position = self.X[self.X.shape[0]-1]
            x = np.linspace(current_position[0] - 0.01, current_position[0] + 0.01, w)
            y = np.linspace(current_position[1] - 0.01, current_position[1] + 0.01, w)
            z = np.linspace(current_position[2] - 0.01, current_position[2] + 0.01, w)


            X_ = self.X[np.where((self.X[:, 0] > current_position[0]-0.02) & (self.X[:, 0] < current_position[0]+0.02)\
                 & (self.X[:, 1] > current_position[1]-0.02) & (self.X[:, 1] < current_position[1]+0.02))]
            Y_ = self.Y[np.where((self.X[:, 0] > current_position[0]-0.02) & (self.X[:, 0] < current_position[0]+0.02)\
                 & (self.X[:, 1] > current_position[1]-0.02) & (self.X[:, 1] < current_position[1]+0.02))]

            print X_
            # print Y_

            fp = Path_Planning(X_, Y_)
            la = fp.decision_func(x,y,z) #ここに範囲を指定した値を入れる
            pred = fp.direction_func()
            pred_normal = pred[0].T[0]
            pred_direction = pred[1].T[0]

            if la == 1: # When convex

                print "-----shape is convex-----"

                pred_position = np.array([self.X[self.X.shape[0]-1]])[0] + a * pred_direction
                self.position.data = pred_position
                # self.orientation.data = self.create_orientation(pred_normal)

                self.publisher_position(self.position)
                # self.publisher_orientation(self.orientation)

                self.X = np.vstack((self.X, self.arm_position))

                # print "pred_normal:", pred_normal
                # print "pred_position:",pred_position
                # print "orientation:", self.orientation.data
                # print "X:", self.X
                # print "Y:", self.Y

                if self.force == 0:
                    self.Y = np.vstack((self.Y, [-1]))

                    n = 1
                    while n < 100:
                        normal_position = self.X[self.X.shape[0]-1] + 0.002 * n * pred_normal.T[0]

                        self.position.data = normal_position
                        # self.orientation.data = self.create_orientation(pred_normal)
                        self.publisher_position(self.position)

                        if self.force == 1:
                            self.X = np.vstack((self.X, self.arm_position))
                            self.Y = np.vstack((self.Y, [0]))
                            break

                elif self.force == 1:
                    self.Y = np.vstack((self.Y, [0]))



            elif la == 0: # When concave

                print "-----shape is concave-----"

                normal_position = self.X[self.X.shape[0]-1] - 0.02 * pred_normal.T[0]
                self.position.data = normal_position
                self.publisher_position(self.position)
                self.X = np.vstack((self.X, self.arm_position))
                self.Y = np.vstack((self.Y, [-1]))

                n = 1
                while n < 50:
                    new_position = self.X[self.X.shape[0]-1] + n * 0.002 * pred_direction
                    self.position.data = new_position
                    self.publisher_position(self.position)

                    if self.force == 1:
                        self.X = np.vstack((self.X, self.arm_position))
                        self.Y = np.vstack((self.Y, [0]))
                        break

                if self.force == 0:
                    n = 1
                    while n < 100:
                        normal_position = self.X[self.X.shape[0]-1] + 0.002 * n * pred_normal.T[0]

                        self.position.data = normal_position
                        # self.orientation.data = self.create_orientation(pred_normal)
                        self.publisher_position(self.position)

                        if self.force == 1:
                            self.X = np.vstack((self.X, self.arm_position))
                            self.Y = np.vstack((self.Y, [0]))
                            break


            time.sleep(0.01)
            rospy.sleep(0.1)
        # # np.save("X", self.X)
        # # np.save("Y", self.Y)
        #
        self.stop_sim()


if __name__=='__main__':
    VU = VREP_UR5()
    VU.main()
