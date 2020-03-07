# coding: utf-8
import rospy
import moveit_commander
import geometry_msgs.msg

rospy.init_node('ur5_ik_velo', anonymous=True, disable_signals=True)
robot = moveit_commander.RobotCommander()   #ロボット全体に対するインタフェース
manipulator = moveit_commander.MoveGroupCommander('manipulator')        #MoveGroupCommanderは特定のグループのための単純なコマンドの実行を行うクラス

manipulator.set_max_acceleration_scaling_factor(0.3)
manipulator.set_max_velocity_scaling_factor(0.3)

#pad_sub = rospy.Subscriber('GamePad', Float32MultiArray, self.pad_callback)

pose = manipulator.get_current_pose() #現在のロボットの姿勢を取得
print pose

def move_func(pose):
    manipulator.set_pose_target(pose)
    plan = manipulator.plan()
    manipulator.execute(plan)

    rospy.sleep(1)
    manipulator.stop()
    manipulator.clear_pose_targets()

def main():
    start_pose = geometry_msgs.msg.Pose()
    target_pose = geometry_msgs.msg.Pose()

    start_pose.position.x = 0.364373887708
    start_pose.position.y = 0.29382045486
    start_pose.position.z = 0.49055705051
    start_pose.orientation.x = 0.493226655903
    start_pose.orientation.y = 0.497155237227
    start_pose.orientation.z = 0.504646164464
    start_pose.orientation.w = 0.504872642056

    target_pose.position.x = 0.46774678195
    target_pose.position.y = 0.0852543972527
    target_pose.position.z = 0.67761829253
    target_pose.orientation.x = 0.493596948291
    target_pose.orientation.y = 0.496169508405
    target_pose.orientation.z = 0.505768161548
    target_pose.orientation.w = 0.504357450952

    pose_list = [start_pose, target_pose]
    t = len(pose_list)
    loop = 3

    for i in range(loop * t):
        move_func(pose_list[i % t])


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
