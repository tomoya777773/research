# coding: utf-8
import rospy
import moveit_commander
import geometry_msgs.msg
import pygame
from pygame.locals import *
import sys
import rospy
from std_msgs.msg import Int8MultiArray

class UR5:
    def __init__(self):
        self.robot = moveit_commander.RobotCommander() #ロボット全体に対するインタフェース
        self.manipulator = moveit_commander.MoveGroupCommander('manipulator') #MoveGroupCommanderは特定のグループのための単純なコマンドの実行を行うクラス

        self.manipulator.set_max_acceleration_scaling_factor(1)
        self.manipulator.set_max_velocity_scaling_factor(1)

        self.pad_sub = rospy.Subscriber('GamePad', Int8MultiArray, self.callback)

        self.pose = self.manipulator.get_current_pose() #現在のロボットの姿勢を取得
        print self.pose

    def move_func(self, pose):
        self.manipulator.set_pose_target(pose)
        plan = self.manipulator.plan()
        self.manipulator.execute(plan)

        rospy.sleep(0.1)
        self.manipulator.stop()
        self.manipulator.clear_pose_targets()


    def callback(self, msg):
        self.pad = msg.data
        # print self.pad

    def main(self):
        self.start_pose = geometry_msgs.msg.Pose()

        self.start_pose.position.x = 0.533680251077
        self.start_pose.position.y = 0.114274167331
        self.start_pose.position.z = 0.363748230537
        self.start_pose.orientation.x = -0.503558007027
        self.start_pose.orientation.y =  0.503066701204
        self.start_pose.orientation.z = 0.497679735051
        self.start_pose.orientation.w = 0.495649179378

        self.move_func(self.start_pose)

        self.target_pose = geometry_msgs.msg.Pose()
        self.target_pose.position.x = 0.533680251077
        self.target_pose.position.y = 0.114274167331
        self.target_pose.position.z = 0.363748230537
        self.target_pose.orientation.x = -0.503558007027
        self.target_pose.orientation.y =  0.503066701204
        self.target_pose.orientation.z = 0.497679735051
        self.target_pose.orientation.w = 0.495649179378


        pose_list = [self.target_pose, self.start_pose]
        t = len(pose_list)
        loop = 1
        # print pose_list[]
        while True:
            print self.pad
            if self.pad[0] == 1:
                self.target_pose.position.x = 0.733680251077
                self.target_pose.position.y = 0.114274167331

                for i in range(loop * t):
                    self.move_func(pose_list[i % t])

            if self.pad[1] == 1:
                self.target_pose.position.x = 0.333680251077
                self.target_pose.position.y = 0.114274167331

                for i in range(loop * t):
                    self.move_func(pose_list[i % t])

            if self.pad[2] == 1:
                self.target_pose.position.x = 0.533680251077
                self.target_pose.position.y = 0.414274167331

                for i in range(loop * t):
                    self.move_func(pose_list[i % t])

            if self.pad[3] == 1:
                self.target_pose.position.x = 0.533680251077
                self.target_pose.position.y = -0.214274167331

                for i in range(loop * t):
                    self.move_func(pose_list[i % t])

            if self.pad[4] == 1:
                self.target_pose.position.x = 0.733680251077
                self.target_pose.position.y = 0.314274167331

                for i in range(loop * t):
                    self.move_func(pose_list[i % t])

            if self.pad[5] == 1:
                self.target_pose.position.x = 0.733680251077
                self.target_pose.position.y = -0.114274167331

                for i in range(loop * t):
                    self.move_func(pose_list[i % t])

            rospy.sleep(0.1)

if __name__ == '__main__':
    rospy.init_node('ur5_ik_velo', anonymous=True, disable_signals=True)
    ur5 = UR5()
    ur5.main()
