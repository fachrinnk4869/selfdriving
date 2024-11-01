#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
import time
import csv
import roslib.packages

path = roslib.packages.get_pkg_dir("hasil")

class FPSLogger:
    def __init__(self, csv_filename):
        self.no = 0
        self.fps_timer = time.time()
        self.fps_interval = 1.0  # Example interval of 1 second
        self.csv_filename = f"{path}/{csv_filename}"
        
        rospy.Subscriber('/fps', Float32, self.fps_callback)

    def fps_callback(self, data):
        fps_data = [self.no, data.data]  # Assuming 'no' is an incrementing index
        self.no += 1
        with open(self.csv_filename, mode='a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(fps_data)
        # print("berhasil")
        print(self.csv_filename)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('fps_logger', anonymous=True)
    
    # Get CSV filename from parameter server
    csv_filename = rospy.get_param('~csv_filename', 'fps_data.csv')
    
    fps_logger = FPSLogger(csv_filename)
    fps_logger.run()
