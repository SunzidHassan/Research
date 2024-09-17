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
import glob
# quaternion transformation
from scipy.spatial.transform import Rotation
import cv2 as cv

#Date time
import datetime

directory = '/home/lw-lab/catkin_ws/src/my_robot_controller/scripts/LLMSim'
os.chdir(directory)

##################################
# Reference files

# mapCoordinates = pd.read_csv('spiral_grid.csv')
mapOrientations = pd.read_csv('orientations.csv')

# olfaction_folder_path = 'Collection22/'
# olfaction_1_csv_path = '*_1_olfaction.csv'
# olfaction_csv_path = '*_olfaction.csv'
grid_path = 'grid.csv'
sensor_template_path = 'Ordered_Olfaction.csv'

xMap = pd.read_csv('x_Map.csv')
yMap = pd.read_csv('y_Map.csv')

sensor_template_path = 'Ordered_Olfaction.csv'
sensor_template = pd.read_csv(sensor_template_path)

##################################
from ultralytics import YOLO
trainedModel = YOLO('/home/sunzid/yolov8/runs/detect/train3/weights/best.pt')

######

from openai import OpenAI
client = OpenAI(api_key='sk-')

##################################

odor_source_instruction = """
You are a plume emitting object detector. You will be provided with a list of objects, and you need to determine if an object in the list can emit plume.
Respond only with the corresponding decision 'Yes' or 'No' without any additional text or punctuation.
If you get an empty list, output 'No' without any additional text or punctuation.
"""

######

LLM_Nav_instruction = """
Your task is to determine the best direction for a mobile robot to move towards an odor source.
The possible directions and their corresponding numerical values are:
- 'Front' = 1
- 'Left' = 2
- 'Back' = 3
- 'Right' = 4

The decision should be based on the following criteria:            
1. Directions with obstacles (Facing_Obstacle = Yes) CANNOT be selected.
2. Directions facing the odor source (Facing_Odor_Source = Yes) SHOULD be prioritized.
3. If there is airflow (Facing_Airflow = Yes), the odor source may be in that direction.
4. High odor concentration (Facing_Odor_Chemical = High) indicates the odor source.

Sensor Data:
"""
LLM_Nav_task = """
Based on this sensor data, classify the best direction for the robot to approach the odor source.
Respond only with the corresponding numerical value: 0, 1, 2, or 3 without any additional text or punctuation.
"""


##################################


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
        
        # Map transition variables
        self.actionID = ""
        self.coordinate_row_id = 4
        self.coordinate_column_ID = 4

        # Set starting position
        self.coordinateX = 4
        self.coordinateY = 4

        olfactionDFData = {
            'Direction': [1, 2, 3, 4],
            'Facing_Obstacle': [None] * 4,
            'Facing_Odor_Source': [None] * 4,
            'Facing_Airflow': [None] * 4,
            'Facing_Odor_Chemical': [None] * 4
        }
        self.olfactionDF = pd.DataFrame(olfactionDFData)


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

    def mapTransition(self):
        if self.actionID == "":
            pass
        elif self.actionID == 1:
            if self.coordinate_row_id > 0:
                self.coordinate_row_id -= 1
        elif self.actionID == 2:
            if self.coordinate_column_ID > 0:
                self.coordinate_column_ID -= 1
        elif self.actionID == 3:
            if self.coordinate_row_id < 4:
                self.coordinate_row_id += 1
        elif self.actionID == 4:
            if self.coordinate_column_ID < 4:
                self.coordinate_column_ID += 1

        self.coordinateX = xMap.iloc[self.coordinate_row_id, self.coordinate_column_ID]
        self.coordinateY = yMap.iloc[self.coordinate_row_id, self.coordinate_column_ID]
        return self.coordinateX, self.coordinateY

    def navGoal(self):
        
        client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"

        if self.navSwitch == 1:
            
            self.xVal = self.coordinateX
            self.yVal = self.coordinateY
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
        if self.time > 5:
            data = ({'time': self.time_list, 'x': self.path_x, 'y': self.path_y, 'z':self.orien_z, 'w': self.orien_w,
                     'plume': self.chemRecord, 'glbWindBlowDir': self.windBlowDirRecord, 'windSpd': self.airFlowRecord})
            self.olfactionDF = pd.DataFrame(data)
            print('Save csv File...')
            self.olfactionDF.to_csv(f'/home/lw-lab/LLMSim/{self.xVal}_{self.yVal}_{self.dirVal}_olfaction.csv', index=False)
            cv.imwrite(f'/home/lw-lab/LLMSim/{self.xVal}_{self.yVal}_{self.dirVal}_vision.png', self.imgDecode)
            if self.coordinateX == 2.4: sys.exit(0)
            self.sensorTable()

    def sensorTable(self):
        self.olfactionDF.iloc[self.dirVal, 1] = sensor_template[(sensor_template['x']==self.coordinateX) & ((sensor_template['y']==self.coordinateY) & (sensor_template['direction']==self.dirVal))]['obstacle']

        visualObjects = trainedModel(self.imgDecode)
        detected_objects = [trainedModel.names[int(box.cls[0])] for box in visualObjects[0].boxes]
        object_counts = {obj: detected_objects.count(obj) for obj in set(detected_objects)}
        object_output = ", ".join([f"{count} {obj}" for obj, count in object_counts.items()])

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
            {"role": "system", "content": odor_source_instruction},
            {"role": "user", "content": object_output}])

        self.olfactionDF.iloc[self.dirVal, 2] = completion.choices[0].message.content.strip()

        self.olfactionDF.iloc[self.dirVal, 3] = (135 < self.localWindDir < 224)
        self.olfactionDF.iloc[self.dirVal, 4] = self.olfactionDF.loc[:,['plume']].mean()

        if self.dirVal == 4: self.LLMNav()
        else: self.nextStep()

    def LLMNav(self):
        self.completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that helps determine the best direction for a mobile robot."},
            {"role": "user", "content": LLM_Nav_instruction + "\n" + self.olfactionDF.to_markdown + "\n" + LLM_Nav_task}
        ])
        # Extract the content part from the completion
        self.actionID = int(self.completion.choices[0].message.content.strip())

        self.coordinateX, self.coordinateY = self.mapTransition()
        self.nextStep()


    def nextStep(self):
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
        else:
            self.loop2Counter += 1
            
        # reset timer
        self.time = 0

        # switch to navigation
        self.navSwitch = 1
        print('-------------------------------------------')
        print('-------------------------------------------')


def main():
    rospy.init_node('LLM_Nav_Node')  # Initializes a node

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