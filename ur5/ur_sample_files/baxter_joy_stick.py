import numpy as np
import sys
import copy

import rospy
from std_msgs.msg import Float32MultiArray
import tf
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

class BAXTERIK:
  def __init__(self):
    print "Setting RobotCommander and MoveGroupCommander ..."
    self.robot = moveit_commander.RobotCommander()
    # self.manipulator = moveit_commander.MoveGroupCommander('manipulator')
    self.manipulator = moveit_commander.MoveGroupCommander('left_arm')

    print "Setting max scaling factors of accel. and vel. ..."
    self.manipulator.set_max_acceleration_scaling_factor(0.3)
    self.manipulator.set_max_velocity_scaling_factor(0.3)

    print "Setting as subscriber ..."
    self.pad_sub = rospy.Subscriber('GamePad', Float32MultiArray, self.pad_callback)

    print "Getting robot's current pose ..."
    self.pose = self.manipulator.get_current_pose()

    self.tf_matrix = np.array([[np.cos(np.pi*3/4), -np.sin(np.pi*3/4), 0],
                               [np.sin(np.pi*3/4),  np.cos(np.pi*3/4), 0],
                               [                0,                  0, 1]])

  def main(self):
    while True:
      self.manipulator.set_pose_target(self.pose)
      plan = self.manipulator.plan()

      norms = []
      joint_values = np.array(self.manipulator.get_current_joint_values())
      plan_joint_values = np.array(plan.joint_trajectory.points[-1].positions)
      for j in plan.joint_trajectory.points:
        plan_joint_values = np.array(j.positions)
        norms.append(np.linalg.norm(joint_values - plan_joint_values))

      print '#'*10
      # print len(plan.joint_trajectory.points)
      # print norms
      print max(norms)
      print '#'*10
      if max(norms) > 1: continue
      self.manipulator.execute(plan)

      rospy.sleep(0.1)
      self.manipulator.stop()
      self.manipulator.clear_pose_targets()

  def pad_callback(self, msg):
    pad = msg.data

    print "Getting robot's current pose(in pad_callback) ..."
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

if __name__=='__main__':
  rospy.init_node('baxter_ik_velo', anonymous=True, disable_signals=True)

  baxter = BAXTERIK()
  baxter.main()
