#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math
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
a=0.01
c = 100
z_limit = 1.02

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

    def predict_func(self):

        """カーネルとカーネルの微分の列ベクトル作成"""
        kernel_x = np.zeros((self.X.shape[0]))
        for i in range(self.zeros.shape[0]):
            kernel_x[i] = self.kernel_func(self.X[i], self.X[self.X.shape[0]-1])  #k(x)
        kernel_x = np.array([kernel_x]).T
        # print "kernel:", kernel_x

        # diff_kernel_x = np.zeros((X.shape[0]))
        diff_kernel_x = np.zeros((self.zeros.shape[0],self.X.shape[1]))
        for i in range(self.zeros.shape[0]):
            for j in range(self.X.shape[1]):
                diff_kernel_x[i][j] = self.diff_kernel_func(self.X[self.X.shape[0]-1], self.X[i], self.X[self.X.shape[0]-1][j], self.X[i][j])

        # print("diff_kernel:", diff_kernel_x)

        """カーネル行列の作成"""
        Kernel_x = np.zeros((self.X.shape[0], self.X.shape[0]))
        for i in range(self.zeros.shape[0]):
            for j in range(self.zeros.shape[0]):
                Kernel_x[i][j] = self.kernel_func(self.X[i], self.X[j])  #K

        # print Kernel_x
        G = Kernel_x + siguma*siguma * np.identity(self.zeros.shape[0])
        invG = np.linalg.inv(G)

        b = np.dot(invG, self.Y - m)
        # print("b:", b)

        """平均と分散"""
        # mean = m + np.dot(kernel_x.T, b)
        # var = 1/length - np.dot(np.dot(kernel_x.T, invG), kernel_x)

        # if self.x[2] < z_limit:
        #     P_limit = - c *(self.x[2] - z_limit)
        # else:
        #     P_limit = 0

        # var_limit = var + P_limit

        """平均と分散の微分"""
        self.diff_mean = (np.dot(b.T, diff_kernel_x)).T

        # if self.x[2] < z_limit:
        #     self.diff_var = (- 2 * np.dot(np.dot(kernel_x.T, invG), diff_kernel_x)).T - np.array([[0], [0], [2 * c * (self.x[2] - z_limit)]])
        # else:
        #     self.diff_var = (- 2 * np.dot(np.dot(kernel_x.T, invG), diff_kernel_x)).T

        # print "11111111111111111111111111",self.diff_var

        self.diff_var = (- 2 * np.dot(np.dot(kernel_x.T, invG), diff_kernel_x)).T
        # print("diff_mean", self.diff_mean)

        """法線"""
        normal = self.diff_mean / np.linalg.norm(self.diff_mean)


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


        return normal, position


class VREP_UR5:
    def __init__(self):
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

        self.position.data = np.array([0, 0, 1.2])
        # self.position.data = np.array([0.45, 0, 1.3])
#
        # self.position.data = np.array([0.01, 0.01, 1.2])
        # self.X = np.array([0.4, -0.03, 0.5])

        self.X = np.array([-0.41, -0.05, 1.29])
        self.Y = np.array([-1])
        # self.force = 0


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

        # beta = 50

        # alpha = 10
        # beta = 358

        #
        # alpha = 180 - (math.atan(normal[1] / normal[2]) / np.pi * 180)
        # beta = - math.atan(normal[0] / normal[2]) / np.pi * 180
        # ganma = math.atan(normal[1] / normal[0]) / np.pi * 180
        ganma = 0
        orientation = np.array([alpha, beta, ganma])

        print "------eurler:", orientation

        orientation = np.array([alpha*np.pi/180, beta*np.pi/180, 0])
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

        """物体に当たるまで直進"""
        while True:
            rospy.sleep(0.1)
            if self.force == 0:
                print "-----not contact-----"

                self.X = np.append(self.X, self.position.data)
                self.Y = np.append(self.Y, -1)

                self.position.data[2] -= 0.01
                self.publisher_position(self.position)


            elif self.force == 1:
                print "-----contact-----"
                # print "aaaaaaaaaaaaaaaa", self.arm_position
                self.X = np.append(self.X, self.arm_position)
                self.Y = np.append(self.Y, 0)

                break



        self.X = np.reshape(self.X, (-1, 3))
        self.Y = np.reshape(self.Y, (-1, 1))

        print self.X
        print self.Y

        """１つ目"""

        fp = Find_position(self.X, self.Y)
        pred = fp.predict_func()
        pred_normal = pred[0].T[0]
        pred_position = pred[1].T[0]

        self.position.data = pred_position
        self.orientation.data = self.create_orientation(pred_normal)

        print "pred_normal:", pred_normal
        print "pred_position:",pred_position
        print "orientation:", self.orientation.data


        self.publisher_position(self.position)
        # rospy.sleep(2)
        self.publisher_orientation(self.orientation)

        self.X = np.vstack((self.X, self.arm_position))

        print self.X
        print self.Y

        cnt = 1
        rospy.sleep(1)

        for i in range(500):

            if self.force == 1:
                self.Y = np.vstack((self.Y, [0]))

                fp = Find_position(self.X, self.Y)
                pred = fp.predict_func()
                pred_normal = pred[0].T[0]
                pred_position = pred[1].T[0]

                self.position.data = pred_position
                self.orientation.data = self.create_orientation(pred_normal)

                print "pred_normal:", pred_normal
                print "pred_position:",pred_position
                print "orientation:", self.orientation.data


                self.publisher_position(self.position)
                # self.publisher_orientation(self.orientation)

                self.X = np.vstack((self.X, self.arm_position))

                print self.X
                print self.Y

                print "-----count:", cnt
                cnt += 1

            elif self.force == 0:
                self.Y = np.vstack((self.Y, [-1]))

                fp = Find_position(self.X, self.Y)
                pred = fp.predict_func()
                pred_normal = pred[0].T[0]


                self.orientation.data = self.create_orientation(pred_normal)
                self.publisher_orientation(self.orientation)

                # self.position.data += 0.001 * pred_normal

                print "pred_normal:", pred_normal
                print "pred_position:", self.position.data
                print "orientation:", self.orientation.data

                while True:
                    if self.force == 1:
                        break
                    elif self.force == 0:
                        self.position.data += 0.01 * pred_normal
                        self.publisher_position(self.position)
                    rospy.sleep(0.1)

                # self.publisher_position(self.position)
                # self.publisher_orientation(self.orientation)

                self.X = np.vstack((self.X, self.arm_position))

                print self.X
                print self.Y

                print "-----count:", cnt
                cnt += 1



            rospy.sleep(0.1)
        np.save("X", self.X)
        np.save("Y", self.Y)

        self.stop_sim()

if __name__ == '__main__':
    VU = VREP_UR5()
    VU.main()
