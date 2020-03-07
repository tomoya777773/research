from GamePad import *
import time
import rospy
from std_msgs.msg import Int8MultiArray
import numpy as np

def shutdown():
    print 'ros shutdown'

def main():
    game_pad = GamePad(0)
    game_pad.button_mode_[0] = 1
    rospy.init_node('GamePad', anonymous=True, disable_signals=True)
    pub = rospy.Publisher('GamePad', Int8MultiArray, queue_size = 30)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(5)

    # button_frag = False

    while True:
        game_pad.Update()

        # if game_pad.buttons_[0] == True:
        #   button_frag = not button_frag

        game_pad_state = Int8MultiArray()
        game_pad_state.data = np.zeros(len(game_pad.buttons_))
        print game_pad_state.data

        # game_pad_state.data = [int(button_frag)] + game_pad.axes_[:3] + list(game_pad.hats_[0])
        # print game_pad_state.data
        for i in range(len(game_pad.buttons_)):
            game_pad_state.data[i] = int(game_pad.buttons_[i])
        print game_pad_state
        pub.publish(game_pad_state)
        rate.sleep()

    rospy.spin()


if __name__=='__main__':
    main()
