#coding: utf-8
import numpy as np
import sys
import copy

import rospy
from std_msgs.msg import Int8
import tf
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

class UR5IK:
  def __init__(self):
    self.robot = moveit_commander.RobotCommander() #ロボット全体に対するインタフェース
    self.manipulator = moveit_commander.MoveGroupCommander('manipulator') #MoveGroupCommanderは特定のグループのための単純なコマンドの実行を行うクラス

    self.manipulator.set_max_acceleration_scaling_factor(0.3)
    self.manipulator.set_max_velocity_scaling_factor(0.3)

    self.pad_sub = rospy.Subscriber('GamePad', Int8, self.pad_callback)

    self.pose = self.manipulator.get_current_pose() #現在のロボットの姿勢を取得

    self.tf_matrix = np.array([[np.cos(np.pi*3/4), -np.sin(np.pi*3/4), 0],
                               [np.sin(np.pi*3/4),  np.cos(np.pi*3/4), 0],
                               [                0,                  0, 1]])

  def main(self):
    while True:
      self.manipulator.set_pose_target(self.pose)
      plan = self.manipulator.plan()

      self.manipulator.execute(plan)

      rospy.sleep(1)
      self.manipulator.stop()
      self.manipulator.clear_pose_targets()

  def pad_callback(self, msg):
    pad = msg.data

    self.pose = self.manipulator.get_current_pose()

    # move_vec = np.array([pad[1]*0.01, -pad[2]*0.01, pad[5]*0.01])
    move_vec = np.array([pad[1]*0.05, -pad[2]*0.05, pad[5]*0.01])
    tf_result = np.dot(self.tf_matrix, move_vec).ravel()
    self.pose.pose.position.x += tf_result[0]
    self.pose.pose.position.y += tf_result[1]
    self.pose.pose.position.z += tf_result[2]

    orie_x = self.pose.pose.orientation.x
    orie_y = self.pose.pose.orientation.y
    orie_z = self.pose.pose.orientation.z
    orie_w = self.pose.pose.orientation.w
    e = list(tf.transformations.euler_from_quaternion((orie_x, orie_y, orie_z, orie_w)))

    e[0] += pad[3]*0.2
    # print e

    q = tf.transformations.quaternion_from_euler(e[0], e[1], e[2])

    self.pose.pose.orientation.x = q[0]
    self.pose.pose.orientation.y = q[1]
    self.pose.pose.orientation.z = q[2]
    self.pose.pose.orientation.w = q[3]

    print self.pose.pose

if __name__=='__main__':
  rospy.init_node('ur5_ik_velo', anonymous=True, disable_signals=True)

  ur5 = UR5IK()
  ur5.main()
