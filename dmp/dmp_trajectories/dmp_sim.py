from dmp_lib import DMPs_discrete
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from transformations import pose_to_list, list_to_pose  # TODO dependency baxter_commander
from tf.transformations import quaternion_about_axis
import numpy as np
import rospy

from std_msgs.msg import Bool, Char, Float32MultiArray, Int32MultiArray

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class DiscreteTaskSpaceTrajectory(object):
    NUMBER_STABLE_STATES = 50

    def __init__(self, init_path, bfs=10):
        assert isinstance(path, Path)
        self.init_path = init_path
        self._dmp = DMPs_discrete(dmps=7, bfs=bfs)
        self._dmp.imitate_path(self._path_to_y_des(init_path, self.NUMBER_STABLE_STATES))

        if self.init_path.header.frame_id == '':
            self.init_path.header.frame_id = self.init_path.poses[0].header.frame_id

    def _path_to_y_des(self, path, nss):
        y_des = []
        for pose_s in path.poses:
            assert pose_s.header.frame_id == self.init_path.header.frame_id
            pose = [val for sublist in pose_to_list(pose_s) for val in sublist]  # flatten to [x, y, z, x, y, z, w]
            y_des.append(pose)

        # Repeat the last point (stable state) n times to avoid brutal cuts due to asymptotic approach
        for n in range(nss):
            y_des.append(y_des[-1])

        return np.array(y_des).transpose()

    def rollout(self, goal):
        assert isinstance(goal, PoseStamped)
        self._dmp.goal = [val for sublist in pose_to_list(goal) for val in sublist]
        y_track, dy_track, ddy_track = self._dmp.rollout()
        return y_track
        # path = Path()
        # for y in y_track:
        #     path.poses.append(list_to_pose([[y[0], y[1], y[2]], [y[3], y[4], y[5], y[6]]],
        #                                    frame_id=self.init_path.header.frame_id))
        # path.header.stamp = rospy.Time.now()
        # path.header.frame_id = self.init_path.header.frame_id
        # return path

class VrepUr5:

    def __init__(self):

        self.start_sim_pub = rospy.Publisher('startSimulation', Bool, queue_size=10)
        self.stop_sim_pub = rospy.Publisher('stopSimulation', Bool, queue_size=10)

        self.position_pub = rospy.Publisher("IKTarget_position",  Float32MultiArray, queue_size=10)
        self.orientation_pub = rospy.Publisher("IKTarget_orientation", Float32MultiArray, queue_size=10)

        # self.position_sub = rospy.Subscriber('arm_position', Float32MultiArray, self.callback_position)
        # self.force_sub = rospy.Subscriber('collisition', Char, self.callback_force)

        self.init_rate = rospy.Rate(20)

        self.position = Float32MultiArray()
        self.orientation = Float32MultiArray()

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

    def main(self, poses):
        self.start_sim()
        rospy.sleep(0.1)

        for pose in poses:
            self.position.data = pose[:3]
            self.publisher_position(self.position)
            rospy.sleep(0.1)
        # self.position.data = poses[0][:3]
        # print self.position
        # self.publisher_position(self.position)



        rospy.sleep(0.2)
        self.stop_sim()

if __name__=='__main__':
    rospy.init_node("test_dmp_traj_discrete")
    path = Path()
    for i in range(100):
        quat = quaternion_about_axis(i*0.00628, (1, 0, 0))
        # pose = list_to_pose([[i/100., i/100., i/100.], quat])
        pose = list_to_pose([[i/100. + 0.4, i/100., i/100. + 1.4], quat])
        pose.header.stamp = i/10.
        path.poses.append(pose)



    goal = list_to_pose([[0.45, 0.2, 1.45], [0, 0, 0, 0]])
    rollout = DiscreteTaskSpaceTrajectory(path).rollout(goal)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(rollout[:, 0], rollout[:, 1], rollout[:,2], c='b')
    # ax.plot(pos_x, pos_y, pos_z, c='orange')

    plt.show()

    vu5 = VrepUr5()
    vu5.main(rollout)
    # vu5.main(rollout.poses)

