#coding: utf-8
import roslib
import rospy
from sensor_msgs.msg import PointCloud2
import numpy
import pylab
import time
import sensor_msgs.point_cloud2 as pc2

def callback(data):
    resolution = (data.height, data.width)

    # 3D position for each pixel
    img = numpy.fromstring(data.data, numpy.float32)

    cloud_points = []
    for p in pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=False):
        cloud_points.append(p[2])

    z_points = numpy.array(cloud_points, dtype=numpy.float32)
    print z_points
    z = z_points.reshape(resolution)
    print z[data.height/2 , data.width/2]

def listener():
    rospy.init_node('subscribe_realsense',anonymous=True)
    rospy.Subscriber('/camera/depth/points', PointCloud2, callback)
    rate = rospy.Rate(1)
    rospy.spin()

if __name__ == "__main__":
    listener()


import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


class SubscribePointCloud(object):
    def __init__(self):
        rospy.init_node('subscribe_custom_point_cloud')
        rospy.Subscriber('/custom_point_cloud', PointCloud2, self.callback)
        rospy.spin()

    def callback(self, point_cloud):
        for point in pc2.read_points(point_cloud):
            rospy.logwarn("x, y, z: %.1f, %.1f, %.1f" % (point[0], point[1], point[2]))
            rospy.logwarn("my field 1: %f" % (point[4]))
            rospy.logwarn("my field 2: %f" % (point[5]))


def main():
    try:
        SubscribePointCloud()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
