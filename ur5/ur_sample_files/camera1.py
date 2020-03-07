#coding: utf-8
import rospy
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2
import numpy as np

class SubscribePointCloud(object):
    def __init__(self):
        rospy.init_node('subscribe_realsense')
        rospy.Subscriber('/camera/depth/image', Image, self.callback)
        rospy.spin()

    def callback(self, point_cloud):

        a = np.fromstring(point_cloud.data, dtype=np.uint8)
        print a, a.shape
def main():
    try:
        SubscribePointCloud()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
