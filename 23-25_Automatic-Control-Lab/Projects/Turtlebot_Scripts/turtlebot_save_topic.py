#!/usr/bin/env python3
# ROSPY
import rospy
import sys

# Turtlebot messages
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.msg import SensorState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import OccupancyGrid
# from nav_msgs.msg import Odometry
import tf

# Calculations
import math
import os
import pandas as pd
import numpy as np
# quaternion transformation
from scipy.spatial.transform import Rotation
# plotting
import matplotlib.pyplot as plt
# image data processing
import cv2 as cv

#Date time
import datetime

directory = '/home/lw-lab/Downloads/Rostopic'
os.chdir(directory)

def globalCostmapCapture(OccupancyGrid):
    rate = rospy.Rate(2)

    globalCostMap = pd.DataFrame(OccupancyGrid.data)

    globalCostMap.to_csv('globalCostMap.csv')

    rate.sleep()

def laserCallback(LaserScan):
    laser = pd.DataFrame(LaserScan.ranges)

    laser.to_csv("laser.csv")
    sys.exit(0)



if __name__ == "__main__":
    rospy.init_node('turtlebot3_TopicCapture')
    rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, globalCostmapCapture)
    rospy.Subscriber("/scan", LaserScan, callback=laserCallback)
    rospy.spin()
    # sys.exit(0)


