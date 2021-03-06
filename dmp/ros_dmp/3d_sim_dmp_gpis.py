#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from dmp_lib import DMPs_discrete
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
        Observation data (x,y,z)
    Y : -1, 0, 1
        Label to determine contact in observation data
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

        """load data"""
        self.orbit_position = np.load("../data/circle_r4_36.npy")
        self.X = np.load("../data/surf_sin_unknown_5000.npy")[1:1000, :]
        self.Y = np.zeros((self.X.shape[0], 1))

        self.orbit_position[:, 2] -= 0.03

        self.position.data = np.array([self.orbit_position[:, 0][0], self.orbit_position[:, 1][0], self.orbit_position[:, 2][0]])
        self.force = 0

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.orbit_position[:, 0], self.orbit_position[:, 1], self.orbit_position[:, 2], alpha = 0.5, color = "red")
        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], alpha = 0.5, color = "red")

        plt.show()

        # self.path_x = np.linspace(0.36, 0.44, 100)
        # self.path_y = np.full(len(self.path_x), 0.01)
        # self.path_z = np.full(len(self.path_x), 1.32)
        # self.paths = np.column_stack([self.path_x, self.path_y, self.path_z])
        # self.position.data = np.array([self.path_x[0], self.path_y[0], self.path_z[0]])

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

    def calculate_quaternion(self, alpha, axis):
        alpha_half = alpha / 2
        cosa = np.cos(alpha_half)
        sina = np.sin(alpha_half)
        v = sina * (axis / np.linalg.norm(np.array(axis, copy=False)))
        return Quat([v[0], v[1], v[2], cosa])

    def contact_object(self, dy, direction):
        beta = 0.6
        gamma = 10
        p = np.zeros(3)
        if np.linalg.norm(dy) > 1e-5:
            cos_phi = np.inner(dy, direction) / (np.linalg.norm(dy)*np.linalg.norm(direction))
            phi = np.arccos(cos_phi)
            cross = np.cross(dy, direction)
            q1 = self.calculate_quaternion(phi, cross)
            R = q1.transform

            print "phi:", math.degrees(phi)
            # print "cross:", cross
            # print "R:", R
            # phi = 10
            dphi = gamma * phi * np.exp(-beta / abs(phi))
            p = np.nan_to_num(np.dot(R, dy) * dphi )
            # p = np.nan_to_num(np.dot(R, dy))

        return p

    def Hesse_matrix(self, po_list):
        if len(po_list) < 5:
            return np.array([1,-1])
        regression_func = lambda param,x,y,z: z - (param[0]*x**2 + param[1]*y**2 + param[2]*x + param[3]*y + param[4])
        param = [0, 0, 0, 0, 0]
        # print "po_list:",po_list
        optimized_param =  scipy.optimize.leastsq(regression_func, param, args=(po_list[:,0], po_list[:,1], po_list[:,2]))[0]

        x = Symbol('x')
        y = Symbol('y')

        z = optimized_param[0]*x**2 + optimized_param[1]*y**2 + optimized_param[2]*x + optimized_param[3]*y + optimized_param[4]

        hesse00 = np.array(diff(diff(z, x), x), dtype=np.float32)
        hesse01 = np.array(diff(diff(z, y), x), dtype=np.float32)
        hesse10 = hesse01.copy()
        hesse11 = np.array(diff(diff(z, y), y), dtype=np.float32)

        Hesse_matrix = np.array([[hesse00,hesse01], [hesse10,hesse11]])

        eigenvalue,v = np.linalg.eig(Hesse_matrix)

        return eigenvalue

    def main(self):

        self.start_sim()
        self.publisher_position(self.position)
        rospy.sleep(1)

        """"Create GPIS """
        gpis = GaussianProcessImplicitSurface(self.X, self.Y)

        """Create DMPs"""
        dmp = DMPs_discrete(dmps=3, bfs=100, dt= 0.01)
        # dmp.imitate_path(y_des=np.array([self.path_x, self.path_y, self.path_z]))
        dmp.imitate_path(y_des=np.array([self.orbit_position[:,0], self.orbit_position[:,1], self.orbit_position[:,2]]))
        # print "goal:", dmp.goal[0]

        # # dmp.goal = np.array([0.44, -0.1, 1.32])
        # dmp.goal[0] += 0.00001
        # dmp.goal[1] += 0.0000000000000001

        # print "goal:", dmp.goal

        # for i in range(dmp.timesteps):
        #     y, dy, ddy = dmp.step(tau=1)
        #     print y
        #     self.position.data = y
        #     self.publisher_position(self.position)
        #     rospy.sleep(0.1)

        # print "goal:", dmp.goal

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

        current_position = np.array(self.arm_position)

        y_track = []
        dy_track = []
        dy_norm = []
        step = 1

        while (np.linalg.norm(current_position[:2] - dmp.goal[:2]) > 10e-3) or (step < 50):
            if step == 500:break

            current_position = np.array(self.arm_position)

            y, dy, ddy = dmp.step(tau=1, state_fb=current_position)

            if self.force == 0:
                n,d = gpis.direction_func(current_position, y)
                # print "nnnnnnnnnnnnnnnnnnnn",n
                interval = 10
                dt = -n * 0.001 /interval
                cnt = 1

                while True:
                    self.position.data = current_position + cnt * dt
                    self.publisher_position(self.position)
                    if self.force == 1:
                        break
                    cnt += 1
                    # print "$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    rospy.sleep(0.01)
                y_track.append(np.copy(self.arm_position))

            current_position = np.array(self.arm_position)

            limit_X = self.X[np.where((self.X[:, 0] > current_position[0]-0.015)\
                                    & (self.X[:, 0] < current_position[0]+0.015)\
                                    & (self.X[:, 1] > current_position[1]-0.015)\
                                    & (self.X[:, 1] < current_position[1]+0.015))]

            eigenvalue = self.Hesse_matrix(limit_X)
            print "limit_X:", len(limit_X)
            # print "eigenvalue:", eigenvalue

            n,d = gpis.direction_func(current_position, y, data_number=len(limit_X))

            if eigenvalue[0] < 0 and eigenvalue[1] < 0:
                direction = d
            else:
                direction = 0.1 * n + d
                direction /= np.linalg.norm(direction)

            direction *= 0.011*np.linalg.norm(dy)
            # direction *= 0.002

            print "direction:", direction
            print "dy:",np.linalg.norm(dy)


            self.position.data = current_position + direction
            self.publisher_position(self.position)

            dy_track.append(np.copy(dy))

            y_track.append(np.copy(self.arm_position))
            dy_norm.append(np.linalg.norm(dy))

            rospy.sleep(0.1)
            step += 1

        y_track = np.array(y_track)
        dy_track = np.array(dy_track)
        dy_norm = np.array(dy_norm)
        self.stop_sim()

        # np.save("../data/known_y", y_track)
        # np.save("../data/known_dy", dy_norm)


        plt.subplot(4, 1, 1)
        plt.plot(range(len(dy_track[:, 0])), dy_track[:, 0], lw = 2)
        plt.subplot(4, 1, 2)
        plt.plot(range(len(dy_track[:, 0])), dy_track[:, 1], lw = 2)
        plt.subplot(4, 1, 3)
        plt.plot(range(len(dy_track[:, 0])), dy_track[:, 2], lw = 2)
        plt.subplot(4, 1, 4)
        plt.plot(range(len(dy_norm)), dy_norm, lw = 2)


        plt.tight_layout()  # タイトルの被りを防ぐ

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(y_track[:, 0], y_track[:, 1], y_track[:, 2], alpha = 0.5, color = "yellow")
        # ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], alpha = 0.5, color = "blue")
        ax.scatter(self.orbit_position[:, 0], self.orbit_position[:, 1], self.orbit_position[:, 2], alpha = 0.5, color = "red")

        plt.show()

if __name__=='__main__':
    VU = VREP_UR5()
    VU.main()



        # """Start search"""
        # for i in tqdm(range(self.orbit_position.shape[0]-1)):
        #     print "\n"
        #     print "STEP :" , step

        #     const = 1
        #     while const < 20:

        #         current_position = np.array(self.arm_position)

        #         distance = np.linalg.norm(current_position[0:2] - self.orbit_position[i+1][0:2], ord=2)

        #         print "-------------------------------------"
        #         # print "count:", step
        #         # print "current position:", current_position
        #         # print "dy:", dy
        #         # print "orbit:",self.orbit_position[i+1]
        #         # print "distance: ", distance
        #         # print "self distance", self.specified_distance
        #         # print "-----------------------------------"

        #         if distance < self.specified_distance:
        #             break
        #         y, dy, ddy = dmp.step(tau=1)
        #         print "y:", y
        #         # if self.orbit_position[i][0] >= 0.38 and self.orbit_position[i][0] < 0.419 and self.orbit_position[i][1] < -0.025 and self.orbit_position[i][1] > -0.045 and distance < self.a * 10:
        #         #     # if data_number < 35:
        #         #     # if distance < self.a * 10:
        #         #     break

        #         if self.force == 1:
        #             limit_X = self.X[np.where((self.X[:, 0] > current_position[0]-0.015)\
        #                                     & (self.X[:, 0] < current_position[0]+0.015)\
        #                                     & (self.X[:, 1] > current_position[1]-0.015)\
        #                                     & (self.X[:, 1] < current_position[1]+0.015))]

        #             eigenvalue = self.Hesse_matrix(limit_X)
        #             print "limit_X:", len(limit_X)
        #             # print "eigenvalue:", eigenvalue

        #             # if len(limit_X) < 5 and distance < self.specified_distance*5 :
        #             #     break

        #             n,d = gpis.direction_func(current_position, y)

        #             if eigenvalue[0] < 0 and eigenvalue[1] < 0:
        #                 direction = d
        #             else:
        #                 direction = 0.3 * n + d
        #                 # direction /= np.linalg.norm(direction)
        #             print "tangent:", direction

        #             d_judge = True

        #         else:
        #             n,d = gpis.direction_func(current_position, self.orbit_position[i+1])

        #             direction = -n
        #             print "normal:", -n

        #             d_judge = False

        #         direction *= 0.01
        #         judge = True
        #         interval = 10
        #         # dt = y- current_position / interval
        #         dt = direction/interval
        #         # dt =  0.03* direction / interval
        #         n = 1
        #         # dmp.y = current_position

        #         while n <= interval:
        #             self.position.data = current_position + n * dt
        #             self.publisher_position(self.position)

        #             distance = np.linalg.norm(self.position.data[0:2] - self.orbit_position[i+1][0:2], ord=2)
        #             if distance < self.specified_distance:
        #                 break
        #             elif (not d_judge and self.force == 1) or (d_judge and self.force == 0):
        #                 break
        #             n += 1
        #             rospy.sleep(0.01)

        #         # current_position = self.position.data
        #         y_track.append(np.copy(self.arm_position))
        #         rospy.sleep(0.01)
        #         const += 1

        #     step += 1

        # y_track = np.array(y_track)
        # dy_track = np.array(dy_track)

        # self.stop_sim()



        # plt.subplot(3, 1, 1)
        # plt.plot(range(len(dy_track[:, 0])), dy_track[:, 0], lw = 2)
        # plt.subplot(3, 1, 2)
        # plt.plot(range(len(dy_track[:, 0])), dy_track[:, 1], lw = 2)
        # plt.subplot(3, 1, 3)
        # plt.plot(range(len(dy_track[:, 0])), dy_track[:, 2], lw = 2)


        # plt.tight_layout()  # タイトルの被りを防ぐ

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(y_track[:, 0], y_track[:, 1], y_track[:, 2], alpha = 0.5, color = "yellow")
        # # ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], alpha = 0.5, color = "blue")
        # ax.scatter(self.orbit_position[:, 0], self.orbit_position[:, 1], self.orbit_position[:, 2], alpha = 0.5, color = "red")

        # plt.show()


# if __name__=='__main__':
#     VU = VREP_UR5()
#     VU.main()







    # def main(self):

    #     gpis = GaussianProcessImplicitSurface(self.X, self.Y, a=1)

    #     dmp = DmpsGpis(dmps=3, bfs=100, dt= 0.01)
    #     dmp.imitate_path(y_des=np.array([self.path_x, self.path_y, self.path_z]))
    #     # dmp.imitate_path(y_des=np.array([self.orbit_position[:,0], self.orbit_position[:,1], self.orbit_position[:,2]]))
    #     # dy = np.array([10e-4, 10e-4, 10e-4])

    #     dy_normalization = np.zeros_like( self.orbit_position.shape[1])
    #     y_track = []
    #     cnt = 1
    #     current_position = self.position.data
    #     change_direction = False
    #     change_t = 1

    #     self.start_sim()

    #     self.publisher_position(self.position)

    #     rospy.sleep(1)

    #     # """物体に当たるまで直進"""
    #     # while True:
    #     #     if self.force == 0:
    #     #         print "-----not contact-----"
    #     #         self.position.data[2] -= 0.002
    #     #         self.publisher_position(self.position)

    #     #     elif self.force == 1:
    #     #         print "-----contact-----"
    #     #         break

    #     #     rospy.sleep(0.1)

    #     # print "-----start serching-----"
    #     # rospy.sleep(1)

    #     while (np.linalg.norm(current_position[0] - dmp.goal[0]) > 10e-3) or (cnt < 10):
    #         if cnt > 1000: break

    #         current_position = np.array(self.arm_position)

    #         print "-----------------------------------"
    #         print "count:", cnt
    #         print "position:",current_position
    #         # print "dy:", dy

    #         n,d = gpis.direction_func(current_position)
    #         # n,d = gpis.direction_func(current_position, self.orbit_position[cnt])

    #         if self.force == 1:

    #             limit_X = self.X[np.where((self.X[:, 0] > current_position[0]-0.01)\
    #                                     & (self.X[:, 0] < current_position[0]+0.01)\
    #                                     & (self.X[:, 1] > current_position[1]-0.01)\
    #                                     & (self.X[:, 1] < current_position[1]+0.01))]

    #             eigenvalue = self.Hesse_matrix(limit_X)
    #             print "eigenvalue:", eigenvalue

    #             n,d = gpis.direction_func(current_position, data_number=limit_X.shape[0])

    #             if eigenvalue[0] < 0 and eigenvalue[1] < 0: # 上に凸のとき
    #                 direction = d
    #             else:
    #                 direction = 0.1 * n + d

    #             d_judge = True

    #             change_direction = False
    #             change_t = 0
    #             print "nnnnnnnnn:", n

    #         else:
    #             n,d = gpis.direction_func(current_position)

    #             direction = -n
    #             d_judge = False
    #             if change_t == 0:
    #                 change_direction = True
    #             else:
    #                 change_direction = False
    #             change_t = 1
    #             print "normal:", -n

    #         # external_force=self.avoid_obstacles(dy, direction, dmp.goal)
    #         external_force=self.contact_object(dy_normalization, direction)

    #         print "external_force:", external_force
    #         # if cnt != 1:
    #         #     dmp.dy = dmp.dy/np.linalg.norm(dmp.dy)

    #         y, dy, ddy = dmp.step(state_fb=current_position,
    #                               external_force=external_force/1000, change_direction=change_direction)
    #         dy_normalization = dy/np.linalg.norm(dy)

    #         judge = True
    #         interval = 10
    #         dt = (y - current_position) / interval
    #         n = 0
    #         # print "333333333333333333", y-current_position
    #         # print "y:", y
    #         # print "current:", current_position
    #         if not d_judge:
    #             while n < interval:
    #                 self.position.data = current_position + n * dt
    #                 self.publisher_position(self.position)

    #                 if self.force == 1:
    #                     break
    #                 n += 1
    #                 rospy.sleep(0.01)

    #         else:
    #             while n < interval:
    #                 self.position.data = current_position + n * dt
    #                 self.publisher_position(self.position)

    #                 if self.force == 0:
    #                     break
    #                 n += 1
    #                 rospy.sleep(0.01)

    #         # current_position = self.position.data
    #         y_track.append(np.copy(self.arm_position))
    #         cnt += 1

    #         rospy.sleep(0.01)


    #     y_track = np.array(y_track)

    #     self.stop_sim()