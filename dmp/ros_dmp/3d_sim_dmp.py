#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from dmp_lib import DmpsGpis
from gpis import GaussianProcessImplicitSurface
import math
from Quaternion import Quat

import rospy
import scipy.optimize
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Bool, Char, Float32MultiArray, Int32MultiArray
from sympy import *
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D



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
        self.X = np.load("../data1/surf_sin_known_5000.npy")[1:500, :]
        self.Y = np.zeros((self.X.shape[0], 1))

        self.orbit_position = np.load("../data1/circle_r4_36.npy")
        print self.orbit_position[:,0]


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.scatter(self.orbit_position[:, 0], self.orbit_position[:, 1], self.orbit_position[:, 2], alpha = 0.5, color = "red")

        plt.show()

        self.path_x = np.linspace(0.36, 0.44, 100)
        self.path_y = np.full(len(self.path_x), 0.01)
        self.path_z = np.full(len(self.path_x), 1.32)

        self.position.data = np.array([self.orbit_position[:, 0][0], self.orbit_position[:, 1][0], self.orbit_position[:, 2][0]])
        self.force = 0


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



    def avoid_obstacles(self, dy, direction, goal):
        beta = 1
        gamma = 100
        R_halfpi = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                            [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])

        Rx_phi = np.array([[1, 0, 0],
                           [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                           [0, np.sin(np.pi/2), np.cos(np.pi/2)]])


        Ry_phi = np.array([[np.cos(np.pi/2), 0, np.sin(np.pi/2)],
                           [0, 1, 0],
                           [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]])

        Rz_phi = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2), 0],
                           [np.sin(np.pi/2), np.cos(np.pi/2), 0],
                           [0, 0, 1]])

        p = np.zeros(3)
        if np.linalg.norm(dy) > 1e-5:

            phi_dy_x = -np.arctan2(dy[2], dy[1])
            phi_dy_y = -np.arctan2(dy[0], dy[2])
            phi_dy_z = -np.arctan2(dy[1], dy[0])


            Rx_phi = np.array([[1, 0, 0],
                            [0, np.cos(phi_dy_x), -np.sin(phi_dy_x)],
                            [0, np.sin(phi_dy_x), np.cos(phi_dy_x)]])

            Ry_phi = np.array([[np.cos(phi_dy_y), 0, np.sin(phi_dy_y)],
                            [0, 1, 0],
                            [-np.sin(phi_dy_y), 0, np.cos(phi_dy_y)]])

            Rz_phi = np.array([[np.cos(phi_dy_z), -np.sin(phi_dy_z), 0],
                            [np.sin(phi_dy_z), np.cos(phi_dy_z), 0],
                            [0, 0, 1]])

            d_vec = np.dot(Rx_phi, direction)
            d_vec = np.dot(Ry_phi, d_vec)
            d_vec = np.dot(Rz_phi, d_vec)

            phi_x = np.arctan2(d_vec[2], d_vec[1])
            phi_y = np.arctan2(d_vec[0], d_vec[2])
            phi_z = np.arctan2(d_vec[1], d_vec[0])

            dphi_x = gamma * phi_x * np.exp(-beta / abs(phi_x))
            dphi_y = gamma * phi_y * np.exp(-beta / abs(phi_y))
            dphi_z = gamma * phi_z * np.exp(-beta / abs(phi_z))

            p = np.nan_to_num(np.dot(Rx_phi, dy) * dphi_x + np.dot(Ry_phi, dy) * dphi_y + np.dot(Rz_phi, dy) * dphi_z)

        return p



    def make_quat(self, alpha, axis):
        alpha_half = alpha / 2
        cosa = np.cos(alpha_half)
        sina = np.sin(alpha_half)
        v = sina * (axis / np.linalg.norm(np.array(axis, copy=False)))
        return Quat([v[0], v[1], v[2], cosa])

    def contact_object(self, dy, direction, goal):
        beta = 5
        gamma = 200
        p = np.zeros(3)
        if np.linalg.norm(dy) > 1e-5:
            cos_phi = np.inner(dy, direction) / (np.linalg.norm(dy)*np.linalg.norm(direction))
            phi = np.arccos(cos_phi)
            cross = np.cross(dy, direction)
            q1 = self.make_quat(phi, cross)
            R = q1.transform

            print "phi:", math.degrees(phi)
            print "cross:", cross
            print "R:", R
            phi = 10
            dphi = gamma * phi * np.exp(-beta / abs(phi))
            p = np.nan_to_num(np.dot(R, dy) * dphi)
        return p

    def main(self):

        gpis = GaussianProcessImplicitSurface(self.X, self.Y, a=1)

        dmp = DmpsGpis(dmps=3, bfs=500, dt= 0.01)
        # dmp.imitate_path(y_des=np.array([self.path_x, self.path_y, self.path_z]))
        dmp.imitate_path(y_des=np.array([self.orbit_position[:,0], self.orbit_position[:,1], self.orbit_position[:,2]]))
        # dy = np.array([10e-4, 10e-4, 10e-4])
        dy = [0,0,0]
        y_track = []
        cnt = 1
        current_position = self.position.data


        self.start_sim()
        rospy.sleep(0.1)

        # """物体に当たるまで直進"""
        # while True:
        #     if self.force == 0:
        #         print "-----not contact-----"
        #         self.position.data[2] -= 0.002
        #         self.publisher_position(self.position)

        #     elif self.force == 1:
        #         print "-----contact-----"
        #         break

        #     rospy.sleep(0.1)

        # print "-----start serching-----"
        # rospy.sleep(1)
        print "aaaaaaa", abs(current_position[0] - dmp.goal[0])

        while (np.linalg.norm(current_position - dmp.goal) > 10e-4) or (cnt < 10):

            print abs(current_position[0] - dmp.goal[0])

            if cnt > 1000: break

            current_position = np.array(self.arm_position)
            print "-----------------------------------"
            print "count:", cnt
            print "position:",current_position
            print "dy:", dy

            n,d = gpis.direction_func(current_position)

            if self.force == 1:
                direction = n
                d_judge = True
                print "nnnnnnnnn:", n

            else:
                direction = -n
                d_judge = False
                print "normal:", -n

            # external_force=self.avoid_obstacles(dy, direction, dmp.goal)
            external_force=self.contact_object(dy, direction, dmp.goal)

            print "external_force:", external_force

            y, dy, ddy = dmp.step()
            dy = dy/np.linalg.norm(dy)
            judge = True
            interval = 10
            dt = (y - current_position) / interval
            n = 0

            if not d_judge:
                while n < interval:
                    self.position.data = current_position + n * dt
                    self.publisher_position(self.position)

                    if self.force == 1:
                        # self.position.data -= dt
                        # self.publisher_position(self.position)
                        break
                    n += 1
                    rospy.sleep(0.1)

            else:
                while n < interval:
                    self.position.data = current_position + n * dt
                    self.publisher_position(self.position)

                    if self.force == 0:
                        break
                    n += 1
                    rospy.sleep(0.1)
                    print self.position.data


                y_track.append(np.copy(self.arm_position))
            cnt += 1

            rospy.sleep(0.1)
        y_track = np.array(y_track)

        self.stop_sim()


if __name__=='__main__':
    VU = VREP_UR5()
    VU.main()