#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
import random

def radius_publisher():
    rospy.init_node('radius_publisher', anonymous=True)
    pub = rospy.Publisher('radius_of_curvature', Float32, queue_size=10)
    rate = rospy.Rate(0.1)  # 10hz

    while not rospy.is_shutdown():
        # Generate a dummy radius of curvature value
        radius = random.uniform(5.0, 100.0)  # Random value between 5 and 100 meters
        rospy.loginfo("Publishing radius of curvature: %f", radius)
        pub.publish(radius)
        rate.sleep()

if __name__ == '__main__':
    try:
        radius_publisher()
    except rospy.ROSInterruptException:
        pass
