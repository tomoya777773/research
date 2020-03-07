#coding: utf-8
import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic", Image, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/depth/image", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print(e)

        # hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV) # RGB表色系からHSV表色系に変換
        #
        # gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # cv_image3  = cv2.Canny(gray_image, 15.0, 30.0);
        # cv_half_image3 = cv2.resize(cv_image3, (0,0),fx=0.5,fy=0.5);

        cv2.imshow("Edge Image", cv_image)
        cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(data, "mono8"))
        except CvBridgeError as e:
            print(e)

def main(args):
        ic = image_converter()
        rospy.init_node('image_converter', anonymous=True)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
