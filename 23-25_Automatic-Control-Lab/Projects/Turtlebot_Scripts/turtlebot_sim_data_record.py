#!/usr/bin/env python3

"""
Import necessary packages
"""
# ROSPY
import rospy
import sys

# Turtlebot messages
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.msg import SensorState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
# from nav_msgs.msg import Odometry
import tf

# Calculations
import math
import os
import pandas as pd
import numpy as np
# quaternion transformation
from scipy.spatial.transform import Rotation
import cv2 as cv

#Date time
import datetime


"""
Server is the main class that read sensor data, process the data with DNN,
and command the robot.
"""

xVal, yVal, dirVal = input('xVal, yVal, dirVal: ').split()


class Server:
    def __init__(self) -> None:
        # Sensor variables
        self.chemical = 0.0
        self.airFlow = 0.0
        self.localWindDir = 0.0
        self.laser = None

        self.rotDegree = 0.0
        self.rotEulerDegree = 0.0

        # robot operating time
        self.timeRate = 50
        self.time = 0.0

        # save data into lists
        self.path_x = []
        self.path_y = []
        self.orien_z = []
        self.orien_w = []
        self.time_list = []
        self.chemRecord = []
        self.windBlowDirRecord = []
        self.airFlowRecord = []

        self.imgDecode = None


    def sensorCallback(self, msg):
        self.chemical = msg.illumination
        self.airFlow = round(msg.cliff, 4)
        self.localWindDir = msg.sonar


    def imageCapture(self, imageInput):
        image = np.asarray(bytearray(imageInput.data), dtype="uint8")
        self.imgDecode = cv.imdecode(image, cv.IMREAD_COLOR)
        

    def tflistener(self, listener):
        while not rospy.is_shutdown():
            try:
                (self.trans,self.rot) = listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            self.xPosition = round(self.trans[0], 1)
            self.yPosition = round(self.trans[1], 1)
            self.zOrientation = round(self.rot[2], 1)
            self.wOrientation = round(self.rot[3], 1)
            self.sensorConversion()

    
    def sensorConversion(self):
        # convert quarternion to Euler degree
        Rot = Rotation.from_quat([0, 0, self.zOrientation, self.wOrientation])
        self.rotEulerDegree = Rot.as_euler('xyz', degrees=True)[2]

        # convert euler degree to rotation degree
        if self.rotEulerDegree < 0:
            self.rotDegree = -self.rotEulerDegree
        else:
            self.rotDegree = 360 - self.rotEulerDegree

        if self.localWindDir > 180:
            self.localWindDir -= 360

        self.globalWindDir = self.localWindDir - self.rotEulerDegree
        if self.globalWindDir > 180:
            self.globalWindDir -= 360

        self.globalWindBlowDir = round(self.globalWindDir + 180, 4)

        self.printInfo()


    def printInfo(self):
        print('Time: {}'.format(self.time))
        print('xPosition: {}'.format(self.xPosition))
        print('yPosition: {}'.format(self.yPosition))
        print('zOrientation: {}'.format(self.zOrientation))
        print('wOrientation: {}'.format(self.wOrientation))
        print('chem: {}'.format(self.chemical))
        print('Wind blow: {}'.format(self.globalWindBlowDir))
        print('Airflow: {}'.format(self.airFlow))

        print('-------------------------------------------')
        self.tracePath()


    def tracePath(self):
        if self.time >= (1/self.timeRate):   # time interval depends on rospy.rate
            
            # record parameters
            self.time_list.append(round(self.time, 4))
            self.path_x.append(self.xPosition)
            self.path_y.append(self.yPosition)
            self.orien_z.append(self.zOrientation)
            self.orien_w.append(self.wOrientation)
            self.chemRecord.append(self.chemical)
            self.windBlowDirRecord.append(self.globalWindBlowDir)
            self.airFlowRecord.append(self.airFlow)

            self.save_data()

        # update the running time
        self.time += (1/self.timeRate)       # time interval depends on rospy.rate
        rate = rospy.Rate(self.timeRate)
        rate.sleep()
    

    def save_data(self):
        # month = datetime.datetime.now().month
        # day = datetime.datetime.now().day
        # hour = datetime.datetime.now().hour
        # minute = datetime.datetime.now().minute

        if self.time > 5:
            
            data = ({'time': self.time_list, 'x': self.path_x, 'y': self.path_y, 'z':self.orien_z, 'w': self.orien_w,
                     'plume': self.chemRecord, 'glbWindBlowDir': self.windBlowDirRecord, 'windSpd': self.airFlowRecord})
            df = pd.DataFrame(data)
            print('Save csv File...')
            df.to_csv(f'/home/lw-lab/LLMSim/{xVal}_{yVal}_{dirVal}_olfaction.csv', index=False)
            cv.imwrite(f'/home/lw-lab/LLMSim/{xVal}_{yVal}_{dirVal}_vision.png', self.imgDecode)
            sys.exit(0)

def main():
    rospy.init_node('LLMSim_Node')  # Initializes a node

    server = Server()

    # rospy.Subscriber("/odom", Odometry, server.positionCallback)
    rospy.Subscriber("/sensor_state", SensorState, callback=server.sensorCallback)
    rospy.Subscriber("/camera/image/compressed", CompressedImage, server.imageCapture)
    listener = tf.TransformListener()
    server.tflistener(listener)

    rospy.spin()  # Loops infinitely until someone stops the program execution


if __name__ == '__main__':
    try:
        main()

    except rospy.ROSInterruptException:
        pass