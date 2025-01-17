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
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
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

directory = '/home/lw-lab/catkin_ws/src/my_robot_controller/scripts/LLMSim'
os.chdir(directory)
mapCoordinates = pd.read_csv('spiral_grid.csv')
mapOrientations = pd.read_csv('orientations.csv')



"""
Server is the main class that read sensor data, process the data with DNN,
and command the robot.
"""

class Server:
    def __init__(self) -> None:
        self.timeRate = 20
        # Sensor variables
        self.chemical = 0.0
        self.airFlow = 0.0
        self.localWindDir = 0.0
        self.laser = None

        self.rotDegree = 0.0
        self.rotEulerDegree = 0.0

        # robot operating time
        self.time = 0.0

        self.xVal = 0.0
        self.yVal = 0.0
        self.dir = 0.0

        # save data into lists
        self.path_x = []
        self.path_y = []
        self.orien_z = []
        self.orien_w = []
        self.time_list = []
        self.chemRecord = []
        self.windBlowDirRecord = []
        self.airFlowRecord = []

        self.navSwitch = 1

        self.loop1Counter = 0
        self.loop2Counter = 0
        

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

        self.navGoal()


    def navGoal(self):
        
        client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"

        if self.navSwitch == 1:
            
            self.xVal = mapCoordinates.iloc[self.loop1Counter,0]
            self.yVal = mapCoordinates.iloc[self.loop1Counter,1]
            self.zVal = mapOrientations.iloc[self.loop2Counter,1]
            self.wVal = mapOrientations.iloc[self.loop2Counter,2]
            self.dirVal = mapOrientations.iloc[self.loop2Counter,0]
            
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = self.xVal
            goal.target_pose.pose.position.y = self.yVal
            goal.target_pose.pose.orientation.z = self.zVal
            goal.target_pose.pose.orientation.w = self.wVal

            client.send_goal(goal)
            wait = client.wait_for_result()

            if wait:
                print(f"{wait}")
                print(f"{self.xVal}, ', ', {self.yVal}, {self.wVal}. ', ', {self.zVal}")
                if client.get_result():
                    print(f"{client.get_result()}")
                    self.navSwitch = 0

        elif self.navSwitch == 0:
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

            self.saveData()

        # update the running time
        self.time += (1/self.timeRate)       # time interval depends on rospy.rate
        rate = rospy.Rate(self.timeRate)
        rate.sleep()


    def saveData(self):
        # month = datetime.datetime.now().month
        # day = datetime.datetime.now().day
        # hour = datetime.datetime.now().hour
        # minute = datetime.datetime.now().minute

        if self.time > 5:
            
            data = ({'time': self.time_list, 'x': self.path_x, 'y': self.path_y, 'z':self.orien_z, 'w': self.orien_w,
                     'plume': self.chemRecord, 'glbWindBlowDir': self.windBlowDirRecord, 'windSpd': self.airFlowRecord})
            df = pd.DataFrame(data)
            print('Save csv File...')
            df.to_csv(f'/home/lw-lab/LLMSim/{self.xVal}_{self.yVal}_{self.dirVal}_olfaction.csv', index=False)
            cv.imwrite(f'/home/lw-lab/LLMSim/{self.xVal}_{self.yVal}_{self.dirVal}_vision.png', self.imgDecode)

            # reset trajectories
            self.path_x = []
            self.path_y = []
            self.orien_z = []
            self.orien_w = []
            self.time_list = []
            self.chemRecord = []
            self.windBlowDirRecord = []
            self.airFlowRecord = []

            # update direction/coorndate
            if (self.loop2Counter + 1 )> (len(mapOrientations)-1):
                self.loop2Counter = 0
                # end program if finish all coordinates
                if (self.loop1Counter + 1) > (len(mapCoordinates)-1): sys.exit(0)
                else: self.loop1Counter += 1
            else:
                self.loop2Counter += 1
                
            # reset timer
            self.time = 0

            # switch to navigation
            self.navSwitch = 1
            print('-------------------------------------------')
            print('-------------------------------------------')


def main():
    rospy.init_node('LLMSim_data_record_Node')  # Initializes a node

    server = Server()

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