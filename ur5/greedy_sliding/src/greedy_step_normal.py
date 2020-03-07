#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt
import numpy as np
import rospy
import scipy.optimize
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Bool, Char, Float32MultiArray, Int32MultiArray
from sympy import *
from tqdm import tqdm
import random
# from pyquaternion import Quaternion
# import tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    Notes
    ----------
    kernel_x : vector
        inverse-multiquadric kernel
    diff_kernel_x : matrix
        differential kernel_x
    Kernel_x : matrix
        kernel function
    """

    def __init__(self, X, Y):

        """
        Parameters
        ----------
        X : float
            観測点(x,y,z)
        Y : -1, 0, 1
            観測点の接触を判断するラベル
        """

        self.X = X
        self.Y = Y
        num = X.shape[0]


        self.m = 1
        self.length=0.2
        siguma=0.03
        self.c = 100
        self.z_limit = 1.28


        """カーネル行列作成"""
        Kernel_x = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                Kernel_x[i][j] = self.kernel_func(self.X[i], self.X[j])  #K
        Kernel_x = np.array(Kernel_x)
        # print(Kernel_x)

        """G, bを求める"""
        G = Kernel_x + siguma**2 * np.identity(num)
        self.invG = np.linalg.inv(G)
        self.b = np.dot(self.invG, self.Y - self.m)
        # print G.shape
        # print self.invG.shape
        # print("b:", self.b.shape)

    def kernel_func(self, data_position, current_position):
        kernel_x = pow(np.linalg.norm(current_position - data_position, ord=2)**2 + self.length**2 , -1/2)
        return kernel_x

    def diff_kernel_func(self, data_position, current_position):
        diff_kernel_x = - pow(np.linalg.norm(current_position - data_position, ord=2)**2 + self.length**2, -3/2) * (current_position - data_position)
        return diff_kernel_x

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

        kernel_x_ = [self.kernel_func(self.X[i], current_position) for i in range(self.X.shape[0])]
        self.kernel_x_ = np.array(kernel_x_)
        # print("self.kernel_x:", self.kernel_x_)

        mean = self.m + np.dot(self.kernel_x_, self.b)
        var = 1/self.length - np.dot(np.dot(self.kernel_x_,self. invG), self.kernel_x_)

        return mean, var

    """凹凸判定"""
    def decision_func(self, po_list):

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
        x, y, z = po_list[0],  po_list[1], po_list[2]
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


        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # ax.set_xlabel("X-axis")
        # # ax.set_ylabel("Y-axis")
        # # ax.set_zlabel("Z-axis")
        # ax.scatter(mean_x, mean_y, mean_z)
        # plt.show()
        # # plt.pause(.01)
        # # plt.cla()

        regression_func = lambda param,x,y,z: z - (param[0]*x**2 + param[1]*y**2 + param[2]*x + param[3]*y + param[4])
        param = [0, 0, 0, 0, 0]
        optimized_param =  scipy.optimize.leastsq(regression_func, param, args=(mean_x, mean_y, mean_z))[0]

        x = Symbol('x')
        y = Symbol('y')

        z = optimized_param[0]*x**2 + optimized_param[1]*y**2 + optimized_param[2]*x + optimized_param[3]*y + optimized_param[4]

        hesse00 = np.array(diff(diff(z, x), x), dtype=np.float64)
        hesse01 = np.array(diff(diff(z, y), x), dtype=np.float64)
        hesse10 = hesse01.copy()
        hesse11 = np.array(diff(diff(z, y), y), dtype=np.float64)

        Hesse_matrix = np.array([[hesse00,hesse01], [hesse10,hesse11]])

        la,v = np.linalg.eig(Hesse_matrix)


        return la

    """法線方向"""
    def direction_func(self, current_position, orbit_position, data_number):

        """kernel_x_
        Parameters
        ----------

        Returns
        ----------
        normal : vector
            normal against object
        d : vector
            new direction on projection
        """


        a = 0.1
        b = 0.9
        c = 1

        if data_number < 35:
            a = 0.9
            b = 0.1


        """カーネルの微分"""
        diff_kernel_x_ = [self.diff_kernel_func(self.X[i], current_position) for i in range(self.X.shape[0])]
        diff_kernel_x_ = np.array(diff_kernel_x_)
        # print "diff_kernel: ", diff_kernel_x_

        """normal"""
        diff_mean = (np.dot(self.b.T, diff_kernel_x_)).T
        normal = diff_mean / np.linalg.norm(diff_mean, ord=2)
        # print("noemal:", normal)

        """projection"""
        projection_n = np.identity(self.X.shape[1]) - np.dot(normal, normal.T)


        """分散の微分"""
        diff_var =  (- 2 * np.dot(np.dot(self.kernel_x_, self.invG), diff_kernel_x_)).T
        diff_var = - diff_var / np.linalg.norm(diff_var)
        print "diff_var: ", diff_var

        """軌道の微分"""
        orbit_position[2] = current_position[2]
        diff_orbit = (orbit_position - current_position) / np.linalg.norm(current_position - orbit_position, ord=2)
        print "diff_orbit: ", diff_orbit

        """ペナルティの微分"""
        if current_position[2] < self.z_limit:
            diff_penalty = np.array([0, 0, -2 * self.c * (current_position[2] - self.z_limit)])
            diff_penalty = diff_penalty / np.linalg.norm(diff_penalty)
        else:
            diff_penalty = np.zeros(3)

        print "diff_penalty: ", diff_penalty


        """diff object function"""
        diff_object = a * diff_var + b * diff_orbit + c * diff_penalty
        print "diff_object:", diff_object

        """新しい方向"""
        S = np.dot(projection_n, diff_object)
        direction = S / np.linalg.norm(S, ord=2)

        # print"projection : ", projection_n
        # print "SSSSSSSSSSSS: ", S
        # print "normal:", normal
        # print "d:", direction

        return normal, direction


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

        """データを読み込み"""
        # self.X = np.load("../data1/surf_sin_known_5000.npy")
        self.X = np.load("../data1/surf_sin_unknown_5000.npy")
        print self.X.shape
        # self.X[:, 2] = self.X[:, 2] - 0.004

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # ax.set_xlabel("X-axis")
        # ax.set_ylabel("Y-axis")
        # ax.set_zlabel("Z-axis")
        # ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], alpha = 0.5, color = "red")

        # plt.show()

        self.Y = np.zeros((self.X.shape[0], 1))

        self.orbit_position = np.load("../data1/circle_r4_36.npy")
        self.position.data = self.orbit_position[0]
        print self.orbit_position
        self.force = 0
        self.d = 0.007
        self.a = 0.003

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
        # print "-----publish position-----"

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

        """
        after robot contacts object
            1. Judge unevenness(Concave or Convex)
            2. Perform each search
        """

        self.start_sim()
        rospy.sleep(0.1)

        """物体に当たるまで直進"""
        while True:
            if self.force == 0:
                print "-----not contact-----"
                self.position.data[2] -= 0.002
                self.publisher_position(self.position)

            elif self.force == 1:
                print "-----contact-----"
                break

            rospy.sleep(0.1)

        print "-----start serching-----"
        rospy.sleep(1)

        orbit_position = []
        step = 1
        for i in tqdm(range(self.orbit_position.shape[0]-1)):
            print "\n"
            print "STEP :" , step

            const = 1
            while const < 100:
                print "---------------------------------------------------"
                current_position = np.array(self.arm_position)
                orbit_position.append(current_position)
                distance = np.linalg.norm(current_position[0:2] - self.orbit_position[i+1][0:2], ord=2)

                # print "current---", current_position
                # print "position-----------", self.orbit_position[i+1][0:2]

                print "distance: ", distance

                w = 10
                x = np.linspace(current_position[0] - 0.005, current_position[0] + 0.005, w)
                y = np.linspace(current_position[1] - 0.005, current_position[1] + 0.005, w)
                z = np.linspace(current_position[2] - 0.007, current_position[2] + 0.007, w)
                sample = np.array([x,y,z])

                """current_positionの近くのデータ点を収集"""
                X_ = self.X[np.where((self.X[:, 0] > current_position[0]-0.008) & (self.X[:, 0] < current_position[0]+0.008)\
                    & (self.X[:, 1] > current_position[1]-0.008) & (self.X[:, 1] < current_position[1]+0.008))]
                Y_ = self.Y[np.where((self.X[:, 0] > current_position[0]-0.008) & (self.X[:, 0] < current_position[0]+0.008)\
                    & (self.X[:, 1] > current_position[1]-0.008) & (self.X[:, 1] < current_position[1]+0.008))]

                print "Num of X_: ",X_.shape
                data_number = X_.shape[0]

                print "aaaaaaaaaaaaaaaaaaaaa", self.orbit_position[i]

                if data_number < 35:
                    if self.orbit_position[i][0] >= 0.38 and self.orbit_position[i][0] < 0.419 and self.orbit_position[i][1] < -0.025 and self.orbit_position[i][1] > -0.045 and distance < self.a * 10:

                    # if distance < self.a * 10:
                       break

                elif distance < self.a :
                    break

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')

                # ax.set_xlabel("X-axis")
                # ax.set_ylabel("Y-axis")
                # ax.set_zlabel("Z-axis")
                # ax.scatter(current_position[0], current_position[1], current_position[2], s = 100)
                # ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], alpha = 0.5, color = "red")
                # ax.scatter(X_[:, 0], X_[:, 1], X_[:, 2], color = "blue")

                # plt.show()

                fp = Path_Planning(X_, Y_)
                eigenvalue = fp.decision_func(sample)
                print "eeeeeeeeeee: ", eigenvalue
                la = 0
                if eigenvalue[0] < 0 and eigenvalue[1] < 0: # 上に凸のとき
                    la = 1

                pred = fp.direction_func(current_position, self.orbit_position[i+1], data_number)
                pred_normal = pred[0].T[0]
                pred_direction = pred[1]
                print "eigenvalue: ", eigenvalue
                print "normal: ", pred[0].T
                print "direction: ", pred[1]

                if la == 1: # When convex

                    print "-----shape is convex-----"
                    pred_position = current_position + self.a * pred_direction
                    self.position.data = pred_position
                    self.publisher_position(self.position)
                    rospy.sleep(0.1)

                    if self.force == 0:

                        n = 1
                        while n < 100:
                            normal_position = pred_position - n * 0.0001 * pred_normal
                            self.position.data = normal_position
                            self.publisher_position(self.position)
                            rospy.sleep(0.1)

                            if self.force == 1:
                                break
                            n += 1



                elif la == 0: # When concave

                    print "-----shape is concave-----"
                    n = 1
                    while n < 100:
                        normal_position = current_position - n * 0.0001 * pred_normal
                        self.position.data = normal_position
                        self.publisher_position(self.position)
                        rospy.sleep(0.1)

                        if self.force == 0:
                            break
                        n += 1

                    n = 1
                    while n < 10:
                        new_position = normal_position + n * self.a * 0.1 * pred_direction
                        self.position.data = new_position
                        self.publisher_position(self.position)
                        rospy.sleep(0.1)

                        if self.force == 1:
                            break
                        n += 1

                    if self.force == 0:
                        n = 1
                        while n < 100:
                            normal_position = new_position + n *  0.0001 * pred_normal
                            self.position.data = normal_position
                            self.publisher_position(self.position)
                            rospy.sleep(0.1)

                            if self.force == 1:
                                break
                            n += 1

                const += 1
                rospy.sleep(0.1)

            step += 1


        # # np.save("X", self.X)
        # # np.save("Y", self.Y)
        #
        np.save("../data1/orbit_position_unknown", orbit_position)
        self.stop_sim()


if __name__=='__main__':
    VU = VREP_UR5()
    VU.main()
