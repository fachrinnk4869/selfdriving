#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32

def publish_angle():
    pub = rospy.Publisher('target_angle', Int32, queue_size=10)
    rospy.init_node('angle_publisher', anonymous=True)
    rate = rospy.Rate(0.2)  # 0.1 Hz (10 seconds)
    
    angle = 490
    while not rospy.is_shutdown():
        rospy.loginfo(f"Publishing target angle: {angle}")
        pub.publish(angle)
        angle += 10
        if angle > 550:
            angle = 480
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_angle()
    except rospy.ROSInterruptException:
        pass
