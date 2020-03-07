#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from time import sleep

import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool


class BaxterGrip:
    def __init__(self):
        rospy.init_node('baxter_grip', disable_signals=True)

        self.start_sim_pub = rospy.Publisher(
            'startSimulation', Bool, queue_size=10)
        self.stop_sim_pub = rospy.Publisher(
            'stopSimulation', Bool, queue_size=10)
        self.ik_target_pub = rospy.Publisher(
            'IKTarget', Float32MultiArray, queue_size=10)
        self.grip_vel_pub = rospy.Publisher('GripVel', Float32, queue_size=10)
        self.init_rate = rospy.Rate(20)
        self.ik_target_sub = rospy.Subscriber(
            'IKCurrent', Float32MultiArray, self.__callback_ik)
        self.cyl_state_sub = rospy.Subscriber(
            'cyl_state', Float32MultiArray, self.__callback_cyl)
        self.hole_state_sub = rospy.Subscriber(
            'hole_state', Float32MultiArray, self.__callback_hole)
        self.right_tip_state = np.zeros(3)
        self.cyl_state = np.zeros(3)
        self.hole_state = np.zeros(3)

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

    def __callback_ik(self, msg):
        self.right_tip_state = np.array(msg.data)

    def __callback_cyl(self, msg):
        self.cyl_state = np.array(msg.data)

    def __callback_hole(self, msg):
        self.hole_state = np.array(msg.data)

    def func_ik(self, pos1, pos2, shake_flag):
        # Float32MultiArray()型のikインスタンス
        ik = Float32MultiArray()

        div = 30

        if shake_flag:
            wait = 3
            srange = 0.0005
        else:
            wait = 1
            srange = 0.0

        for i in range(div):
            for _ in range(wait):
                shake = 2.0 * np.random.rand() - 1.0   # -1 ~ 1
                shake *= srange
                ik.data = pos1 + (pos2 - pos1) * \
                    (float(i + 1) / float(div)) + [shake, shake, 0]
                self.ik_target_pub.publish(ik)
                self.init_rate.sleep()
        sleep(0.2)

    def func_gripVel(self, vel):
        # Float32()型のgripインスタンス
        grip = Float32()
        grip.data = vel

        wait = 20
        for _ in range(wait):
            self.grip_vel_pub.publish(grip)
            self.init_rate.sleep()
        sleep(0.2)

    def main(self):
        self.start_sim()
        print 'start simulation'
        sleep(1)

        gripVel = 0.1

        # ペグ把持フェイズ #################################
        pos_current = self.right_tip_state
        pos_target = self.cyl_state + [0, 0, 0.2]
        self.func_ik(pos_current, pos_target, 0)

        pos_current = pos_target
        pos_target = self.cyl_state + [0, 0, 0.03]
        self.func_ik(pos_current, pos_target, 0)

        self.func_gripVel(gripVel)

        # ペグ挿入フェイズ #################################
        pos_current = pos_target
        pos_target = self.hole_state + [0, 0, 0.15]
        self.func_ik(pos_current, pos_target, 0)

        pos_current = pos_target
        pos_target = self.hole_state + [0, 0, 0.12]
        self.func_ik(pos_current, pos_target, 0)

        pos_current = pos_target
        pos_target = self.hole_state + [0, 0, 0.06]
        self.func_ik(pos_current, pos_target, 1)

        self.func_gripVel(-gripVel)

        pos_current = pos_target
        pos_target = self.hole_state + [0, 0, 0.15]
        self.func_ik(pos_current, pos_target, 0)

        self.func_gripVel(gripVel)

        pos_current = pos_target
        pos_target = self.hole_state + [0, 0, 0.06]
        self.func_ik(pos_current, pos_target, 0)

        pos_current = pos_target
        pos_target = self.hole_state + [0, 0, 0.15]
        self.func_ik(pos_current, pos_target, 0)

        sleep(2)

        self.stop_sim()
        print 'stop simulation'


if __name__ == '__main__':
    bg = BaxterGrip()
    bg.main()
