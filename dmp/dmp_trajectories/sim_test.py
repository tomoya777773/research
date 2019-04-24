#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Bool, Char, Float32MultiArray, Int32MultiArray
import numpy as np

class DmpUR5:

    """
    Attributes
    ----------
    X : float
        観測点(x,y,z)
    Y : -1, 0, 1
        観測点の接触を判断するラベル
    """

    def __init__(self):
        rospy.init_node('ur5_moving', disable_signals=True)

        self.start_sim_pub = rospy.Publisher('startSimulation', Bool, queue_size=10)
        self.stop_sim_pub = rospy.Publisher('stopSimulation', Bool, queue_size=10)

        self.position_pub = rospy.Publisher("IKTarget_position",  Float32MultiArray, queue_size=10)
        self.orientation_pub = rospy.Publisher("IKTarget_orientation", Float32MultiArray, queue_size=10)

        # self.position_sub = rospy.Subscriber('arm_position', Float32MultiArray, self.callback_position)
        # self.force_sub = rospy.Subscriber('collisition', Char, self.callback_force)

        self.init_rate = rospy.Rate(20)

        self.position = Float32MultiArray()
        self.orientation = Float32MultiArray()

        self.position.data = np.array([0.45, 0.2, 1.3])
        print self.position

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

    # def callback_force(self, msg):
    #     self.force = msg.data
    #     # print self.force

    # def callback_position(self, msg):
    #     self.arm_position = msg.data

    def publisher_position(self, position):
        self.position_pub.publish(position)
        # print "-----publish position-----"

    def publisher_orientation(self, orientation):
        self.orientation_pub.publish(orientation)
        print "-----publish orientation-----"

    def main(self):
        self.start_sim()
        # rospy.sleep(0.1)

        # for path in range(self.paths):
        print "11111111111111111111111111"
        self.publisher_position(self.position)
        print "222222222222222222222222222"
        self.stop_sim()


if __name__=='__main__':
    ur5_dmp = DmpUR5()
    ur5_dmp.main()
