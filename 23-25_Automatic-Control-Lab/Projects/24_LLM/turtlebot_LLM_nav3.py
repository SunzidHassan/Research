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
# plotting
import matplotlib.pyplot as plt
# image data processing
import cv2 as cv

from openai import OpenAI
api_key=''

import base64
import requests

#Date time
import datetime

"""
Parameter Initializations
"""
            
# chemical sensor threshold
chmThr = 400

# chmThr = 100

# trajectory
path_x = []
path_y = []


"""
Server is the main class that read sensor data, process the data with DNN,
and command the robot.
"""
class Server:
    def __init__(self) -> None:
        # Source position 1
        # self.source_position_x = float(input('Enter Goal x-Position: '))
        # self.source_position_y = float(input('Enter Goal y-Position: '))
        self.source_position_x = 3.2
        self.source_position_y = 1.4
        self.dist_threshold = 0.8 # Todo: reduce the distance threshold after vision-laser fusion.
        self.dist = 0.0

        # Laser scan range threshold
        self.laserAngleThr0 = 0.75
        self.laserAngleThr2 = 0.5
        self.laserAngleThr3 = 0.3

        # Sensor variables
        self.chemical = 0.0
        self.airFlow = 0.0
        self.localWindDir = 0.0
        self.laser = None

        self.rotDegree = 0.0
        self.rotEulerDegree = 0.0

        # laser variables
        self.laserAngleFront_0, self.laserAngleSLeft_15, self.laserAngleLeft_60, self.laserAngleBack_120, self.laserAngleRight_180, self.laserAngleSRight_225 = 0, 15, 60, 120, 180, 225

        self.obstacle_boolFront = True
        self.obstacle_boolSlightRgtLft = True
        self.obstacle_boolRgtLft = True
        self.obstacleBool = False

        self.targetHeading = True
        self.loop = 0

        # Vision variables
        self.sequence = None
        self.imgDecode = None

        # Behavior variables
        self.behavior_switch = 'switch_to_out'
        self.Sign = 1
        self.findDir = 0

        # target heading in degrees and global frame
        self.target_heading = 0.0

        # robot operating time
        self.timeRate = 30
        self.time = 0.0

        # chemical plume detection
        # self.detection_flag = 0             # 0 for non-detection, 1 for detection
        # self.pre_chem = 0.0

        # dnn parameters
        self.ux = 0.0
        self.uy = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.plume_non_detection = 0.0

        # save data into lists
        self.path_x = []
        self.path_y = []
        self.actionIDRecord = []
        self.ux_list = []
        self.uy_list = []
        self.vx_list = []
        self.vy_list = []
        self.time_list = []
        self.detection_flag_list = []
        self.behavior_switch_list = []
        self.behavior_list = []
        self.plume_non_detection_list = []

        # Move parameter
        self.move = 0.0
        self.actionID = 5

    def imageCapture(self, imageInput):
        image = np.asarray(bytearray(imageInput.data), dtype="uint8")
        self.imgDecode = base64.b64encode(image).decode('utf-8')
        self.sequence = imageInput.header.seq

        self.LLMNav()

    def sensorCallback(self, msg):
        # self.chemical = 0
        self.chemical = msg.illumination
        self.airFlow = msg.cliff
        self.localWindDir = msg.sonar

       
    def laserCallback(self, msg):
        self.laser = msg


    def tflistener(self, listener):
        while not rospy.is_shutdown():
            try:
                (self.trans,self.rot) = listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            self.xPosition = self.trans[0]
            self.yPosition = self.trans[1]
            self.zOrientation = self.rot[2]
            self.wOrientation = self.rot[3]
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

        self.prelocalWindDir = self.localWindDir
        self.globalWindDir = self.localWindDir - self.rotEulerDegree
        if self.globalWindDir > 180:
            self.globalWindDir -= 360

        # # temp sensor fixing
        # self.globalWindDir = 0

        self.globalWindBlowDir = self.globalWindDir + 180
        self.setSensorFlag()


    def LLM4o(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.robot_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.imgDecode}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        response = int(requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()['choices'][0]['message']['content'])
        return response
        


    def LLMNav(self):
        if (self.sequence < 10) or (self.sequence % 180 == 0):
            cv.imwrite(f'/home/lw-lab/LLMSim/{self.xPosition}_{self.yPosition}_{self.chemical}_vision.png', self.imgDecode)
            self.robot_prompt = f"""
            Task: Determine the best direction for a mobile robot to move towards an odor source.

            Input image: The given image is the robot's front view.
            Input data: "current odor concentration": {self.chemical}.

            Action Selection Instruction 1: If there is an vapor emiting object in the image, it is likely the odor source. If the likely odor source is in the center of the image, the robot should move forward to approach it. (Action = 1).
            Action Selection Instruction 2: If there is an vapor emiting object in the image, it is likely the odor source. If the likely odor source is in the right part of the image, the robot should move right to approach it. (Action = 2)
            Action Selection Instruction 3: If there is an vapor emiting object in the image, it is likely the odor source. If the likely odor source is in the left part of the image, the robot should move left to approach it. (Action = 3)
            Action Selection Instruction 4: If there is no object that can emit vapor in the image, and if "current odor concentration" is greater than {self.chmThr}, then move upwind to approach the odor source. (Action = 4)
            Action Selection Instruction 5: If there is no object that can emit vapor in the image, and if "current odor concentration" is less than {self.chmThr}, move crosswind. (Action = 5)

            Output Instruction 1: If there is a vapor-emiting object in the image, selection one of the actions 1, 2 or 3.
            Output Instruction 2: If there is no vapor-emiting object in the image, select one of the actions 4 or 5.
            Output Instruction 3: Respond with the corresponding numerical value of the action (1, 2, 3, 4, or 5) without any additional text or punctuation."""
            self.actionID = self.LLM4o()

    def check_behavior(self, behavior_switch):
        # default behavior is zigzag
        behavior = 'Out'
        if behavior_switch == 'switch_to_obstacleAvoid':
            behavior = 'ObstacleAvoidance'
        elif behavior_switch == 'switch_to_vision_front':
            behavior = 'Vision_front'
        elif behavior_switch == 'switch_to_vision_right':
            behavior = 'Vision_right'
        elif behavior_switch == 'switch_to_vision_left':
            behavior = 'Vision_left'
        elif behavior_switch == 'switch_to_in':
            behavior = 'In'
        elif behavior_switch == 'switch_to_out':
            behavior = 'Out'
        return behavior


    def pid_heading(self, target_heading, current_heading):
        kp = 0.03
        error = current_heading - target_heading
        if error < -180:
            error += 360
        if error > 180:
            error -= 360
        pid_output = kp * error
        if pid_output > 0.3:
            pid_output = 0.3
        if pid_output < -0.3:
            pid_output = -0.3
        return -pid_output
    

    def setSensorFlag(self):
        self.laserAngleFront_0, self.laserAngleSLeft_15, self.laserAngleSLeft_30, self.laserAngleLeft_60, self.laserAngleBack_120, self.laserAngleRight_180, self.laserAngleSRight_210, self.laserAngleSRight_225 = 0, 15, 30, 60, 120, 180, 210, 225

        self.obstacle_boolFront = self.laser.ranges[self.laserAngleFront_0] < self.laserAngleThr0
        self.obstacle_boolSlightRgtLft = (self.laser.ranges[self.laserAngleSLeft_15] < self.laserAngleThr0) or (self.laser.ranges[self.laserAngleSRight_225] < self.laserAngleThr0)
        self.obstacle_boolHalfRgtLft = (self.laser.ranges[self.laserAngleSLeft_30] < self.laserAngleThr2) or (self.laser.ranges[self.laserAngleSRight_210] < self.laserAngleThr2)
        self.obstacle_boolRgtLft = (self.laser.ranges[self.laserAngleLeft_60] < self.laserAngleThr3) or (self.laser.ranges[self.laserAngleRight_180] < self.laserAngleThr3)
        # self.obstacle_boolSlightRgtLft = (0.01 < self.laser.ranges[self.laserAngleSLeft_15] < self.laserAngleThr2) or (0.01 < self.laser.ranges[self.laserAngleSRight_225] < self.laserAngleThr2)
        # self.obstacle_boolRgtLft = (0.01 < self.laser.ranges[self.laserAngleLeft_60] < self.laserAngleThr3) or (0.01 < self.laser.ranges[self.laserAngleRight_180] < self.laserAngleThr3)

        self.obstacleBool = self.obstacle_boolFront or self.obstacle_boolSlightRgtLft or self.obstacle_boolHalfRgtLft or self.obstacle_boolRgtLft
        # if laser obstacle detected
        if self.obstacleBool: self.detection_flag = 'obstacle_detected'

        # determine whether the robot detects obstacle or plume source object in vision
        elif self.actionID == 1: self.detection_flag = 'vision_detected_front'
        elif self.actionID == 2: self.detection_flag = 'vision_detected_right'
        elif self.actionID == 3: self.detection_flag = 'vision_detected_left'

        # if object is not detected in vision, trace plume
        elif self.actionID == 4: self.detection_flag = 'chemical_detected'
        elif self.actionID == 5: self.detection_flag = 'chemical_not_detected'

        self.navigate()


    def navigate(self):
        self.move = Twist()  # Creates a Twist message type object
        rate = rospy.Rate(self.timeRate)

        # determine the current behaviors
        behavior = self.check_behavior(behavior_switch=self.behavior_switch)

        # track-in behavior
        if behavior == 'In':
            self.targetHeading = True
            if self.detection_flag == 'chemical_detected':
                self.target_heading = self.globalWindDir
            elif self.detection_flag == 'obstacle_detected':
                self.behavior_switch = 'switch_to_obstacleAvoid'
            elif self.detection_flag == 'vision_detected_front':
                self.behavior_switch = 'switch_to_vision_front'
            elif self.detection_flag == 'vision_detected_right':
                self.behavior_switch = 'switch_to_vision_right'
            elif self.detection_flag == 'vision_detected_left':
                self.behavior_switch = 'switch_to_vision_left'
            elif self.detection_flag == 'chemical_not_detected':
                self.behavior_switch = 'switch_to_out'


        # track-out behavior
        if behavior == 'Out':
            self.targetHeading = True
            # move across wind direction
            if self.detection_flag == 'chemical_not_detected':
                self.target_heading = self.globalWindDir + 90
            elif self.detection_flag == 'obstacle_detected':
                self.behavior_switch = 'switch_to_obstacleAvoid'
            elif self.detection_flag == 'vision_detected_front':
                self.behavior_switch = 'switch_to_vision_front'
            elif self.detection_flag == 'vision_detected_right':
                self.behavior_switch = 'switch_to_vision_right'
            elif self.detection_flag == 'vision_detected_left':
                self.behavior_switch = 'switch_to_vision_left'
            elif self.detection_flag == 'chemical_detected':
                self.behavior_switch = 'switch_to_in'


        # vision-led behavior
        if behavior == 'Vision_front':
            self.targetHeading = False
            if self.detection_flag == 'vision_detected_front':
                self.move.linear.x = 0.08
                self.move.angular.z = 0.0
            if self.detection_flag == 'obstacle_detected':
                self.behavior_switch = 'switch_to_obstacleAvoid'
            elif self.detection_flag == 'vision_detected_right':
                self.behavior_switch = 'switch_to_vision_right'
            elif self.detection_flag == 'vision_detected_left':
                self.behavior_switch = 'switch_to_vision_left'
            elif self.detection_flag == 'chemical_not_detected':
                self.behavior_switch = 'switch_to_out'
            elif self.detection_flag == 'chemical_detected':
                self.behavior_switch = 'switch_to_in'


        # vision-led behavior
        if behavior == 'Vision_right':
            self.targetHeading = False
            if self.detection_flag == 'vision_detected_right':
                self.move.linear.x = 0.08
                self.move.angular.z = 0.03
            if self.detection_flag == 'obstacle_detected':
                self.behavior_switch = 'switch_to_obstacleAvoid'
            elif self.detection_flag == 'vision_detected_front':
                self.behavior_switch = 'switch_to_vision_front'
            elif self.detection_flag == 'vision_detected_left':
                self.behavior_switch = 'switch_to_vision_left'
            elif self.detection_flag == 'chemical_not_detected':
                self.behavior_switch = 'switch_to_out'
            elif self.detection_flag == 'chemical_detected':
                self.behavior_switch = 'switch_to_in'

        # vision-led behavior
        if behavior == 'Vision_left':
            if self.detection_flag == 'vision_detected_left':
                self.targetHeading = False
                self.move.linear.x = 0.08
                self.move.angular.z = -0.03
            elif self.detection_flag == 'obstacle_detected':
                self.behavior_switch = 'switch_to_obstacleAvoid'
            elif self.detection_flag == 'vision_detected_front':
                self.behavior_switch = 'switch_to_vision_front'
            elif self.detection_flag == 'vision_detected_right':
                self.behavior_switch = 'switch_to_vision_right'
            elif self.detection_flag == 'chemical_detected':
                self.behavior_switch = 'switch_to_in'
            elif self.detection_flag == 'chemical_not_detected':
                self.behavior_switch = 'switch_to_out'

        # obstacle-avoid behavior
        if behavior == 'ObstacleAvoidance':
            if self.detection_flag == 'obstacle_detected':
                thr1 = 0.8 # Laser scan range threshold
                thr2 = 0.6
                # no obstacles front, slightly left or slightly right
                if self.laser.ranges[0]>thr1 and self.laser.ranges[25]>thr2 and self.laser.ranges[215]>thr2:
                    self.targetHeading = True
                    self.loop = 1
                    if self.laser.ranges[25] > self.laser.ranges[215]:
                        self.target_heading = self.globalWindDir + 90
                    else:
                        self.target_heading = self.globalWindDir - 90
                    # self.move.linear.x = 0.05 # go forward (linear velocity)
                    # self.move.angular.z = 0.0 # do not rotate (angular velocity)
                # no obstacles slightly left
                elif self.laser.ranges[25] > thr1:
                    self.targetHeading = False
                    self.move.linear.x = 0.0
                    self.move.angular.z = 0.5
                    self.loop = 2
                    if self.laser.ranges[0]>thr1 and self.laser.ranges[25]>thr2 and self.laser.ranges[215]>thr2:
                        self.targetHeading = True
                        self.move.linear.x = 0.08
                        self.move.angular.z = 0.0
                        self.loop = 3
                        self.target_heading = self.globalWindDir + 90
                # no obstacles slightly right
                elif self.laser.ranges[215] > thr1:
                    self.targetHeading = False
                    self.move.linear.x = 0.0
                    self.move.angular.z = -0.5
                    self.loop = 4
                    if self.laser.ranges[0]>thr1 and self.laser.ranges[25]>thr2 and self.laser.ranges[215]>thr2:
                        self.targetHeading = True
                        self.move.linear.x = 0.08
                        self.move.angular.z = 0.0
                        self.loop = 5
                        self.target_heading = self.globalWindDir - 90
                # obstacle in both sligtly left and right and front
                else:
                    if self.laser.ranges[25] > self.laser.ranges[215]:
                        self.targetHeading = False
                        self.move.linear.x = 0.0
                        self.move.angular.z = 0.5
                        self.loop = 6
                        if self.laser.ranges[0]>thr1 and self.laser.ranges[25]>thr2 and self.laser.ranges[215]>thr2:
                            self.targetHeading = True
                            self.move.linear.x = 0.08
                            self.move.angular.z = 0.0
                            self.loop = 7
                            self.target_heading = self.globalWindDir + 90
                    # if space in slight right is greater than slight left
                    elif self.laser.ranges[25] < self.laser.ranges[215]:
                        self.targetHeading = False
                        self.move.linear.x = 0.0
                        self.move.angular.z = -0.5
                        self.loop = 8
                        if self.laser.ranges[0]>thr1 and self.laser.ranges[25]>thr2 and self.laser.ranges[215]>thr2:
                            self.targetHeading = True
                            self.move.linear.x = 0.08
                            self.move.angular.z = 0.0
                            self.loop = 9
                            self.target_heading = self.globalWindDir - 90

            elif self.detection_flag == 'vision_detected_front':
                self.behavior_switch = 'switch_to_visionNav_front'
            elif self.detection_flag == 'vision_detected_right':
                self.behavior_switch = 'switch_to_visionNav_right'
            elif self.detection_flag == 'vision_detected_left':
                self.behavior_switch = 'switch_to_visionNav_left'
            elif self.detection_flag == 'chemical_detected':
                self.behavior_switch = 'switch_to_in'
            elif self.detection_flag == 'chemical_not_detected':
                self.behavior_switch = 'switch_to_out'

        # PID heading control after the heading is obtained
        if self.targetHeading == True:
            if self.target_heading > 180:
                self.target_heading -= 360
            if self.target_heading < -180:
                self.target_heading += 360
            self.move.angular.z = self.pid_heading(target_heading=self.target_heading, current_heading=self.rotEulerDegree)
            self.move.linear.x = 0.08

        pub.publish(self.move) # publish the move object
        self.printInfo()
        rate.sleep()
    

    def printInfo(self):
        behavior = self.check_behavior(behavior_switch=self.behavior_switch)

        print('Laser reading in front: {}, front threshold{}'.format(self.laser.ranges[self.laserAngleFront_0], self.laserAngleThr0))
        print('Laser reading slight Left: {}, slight Right: {}, threshold: {}'.format(round(self.laser.ranges[self.laserAngleSLeft_15],1), round(self.laser.ranges[self.laserAngleSRight_225],1), self.laserAngleThr2))
        print('Laser reading Left: {}, Right: {}, threshold: {}'.format(round(self.laser.ranges[self.laserAngleLeft_60],1), round(self.laser.ranges[self.laserAngleRight_180],1), self.laserAngleThr3))
        print('Obstacle bool front: {}, slight right-left: {}, right-left: {}, obstacle condition: {}'.format(self.obstacle_boolFront, self.obstacle_boolSlightRgtLft, self.obstacle_boolRgtLft, self.obstacleBool))
        print('Obstacle loop: {}'.format(self.loop))
        print('**')
        print('Time: {}'.format(self.time))
        print('x-Position: {}'.format(round(self.xPosition,1)))
        print('y-Position: {}'.format(round(self.yPosition, 1)))
        # print('rotEulerDegree: {}'.format(self.rotEulerDegree))
        print('rotDegree: {}'.format(round(self.rotDegree, 1)))
        print('**')
        print('Chemical Sensor: {} ppm'.format(round(self.chemical, 1)))
        print('Airflow Speed: {} m/s'.format(round(self.airFlow, 1)))
        print('Local Airflow Direction: {} degrees'.format(round(self.localWindDir, 1)))
        # print('Global Airflow Blow Direction: {} degrees'.format(self.globalWindBlowDir))
        print('Global Airflow Direction: {} degrees'.format(round(self.globalWindDir, 1)))
        #print('Avg Global Airflow Direction: {} degrees'.format(self.AvgGlobalWindDir))
        print('**')
        print('Current Action: {}'.format(self.actionID))
        print('Current Detection Flag: {}'.format(self.detection_flag))
        print('Current Behavior Switch: {}'.format(self.behavior_switch))
        print('Current Behavior: {}'.format(behavior))
        print('Target Heading: {}'.format(round(self.target_heading, 1)))
        print('Current Heading: {}'.format(round(self.rotEulerDegree, 1)))
        print('Current linear speed: {}'.format(self.move.linear.x))
        print('Current angualr speed: {}'.format(self.move.angular.z))
        print('**')
        print('Distance to the Source: {}'.format(round(self.dist, 1)))
        print('-------------------------------------------')
        self.tracePath()


    def tracePath(self):
        if self.time >= (1/self.timeRate):   # time interval depends on rospy.rate
            
            # record parameters
            self.time_list.append(round(self.time, 4))
            self.path_x.append(self.xPosition)
            self.path_y.append(self.yPosition)
            # self.chemRecord.append(self.chemical)
            # self.windBlowDirRecord.append(self.globalWindBlowDir)
            # self.airFlowRecord.append(self.airFlow)
            self.behavior_list.append(self.check_behavior(behavior_switch=self.behavior_switch))
            self.actionIDRecord.append(self.actionID)

            self.saveData()

        # update the running time
        self.time += (1/self.timeRate)       # time interval depends on rospy.rate
        rate = rospy.Rate(self.timeRate)
        rate.sleep()


    def saveData(self):
        month = datetime.datetime.now().month
        day = datetime.datetime.now().day
        hour = datetime.datetime.now().hour
        minute = datetime.datetime.now().minute

        self.dist = math.sqrt((self.xPosition - self.source_position_x)**2 + (self.yPosition - self.source_position_y)**2)
        if self.dist < self.dist_threshold or self.time > 120:
            savePrompt = input('Save the path? [Y/n]: ')
            if savePrompt == 'Y' or savePrompt == 'y':
                data = {'time': self.time_list, 'x': self.path_x, 'y': self.path_y, 'behavior': self.behavior_list, 'action': self.actionIDRecord}
                df = pd.DataFrame(data)
                print('Save csv File...')
                df.to_csv(f'/home/lw-lab/LLMSim/Mn_{month}-Dy_{day}-Hr_{hour}-Mt_{minute}.csv', index=False)
                sys.exit(0)
            else:
                sys.exit(0)


def main():
    rospy.init_node('moth_inspired_with_vision_navigation_node')  # Initializes a node

    server = Server()

    # rospy.Subscriber("/odom", Odometry, server.positionCallback)
    rospy.Subscriber("/sensor_state", SensorState, callback=server.sensorCallback)
    rospy.Subscriber("/scan", LaserScan, callback=server.laserCallback)
    rospy.Subscriber("/camera/image/compressed", CompressedImage, callback=server.imageCapture)
    listener = tf.TransformListener()
    server.tflistener(listener)

    rospy.spin()  # Loops infinitely until someone stops the program execution


if __name__ == '__main__':
    try:
        pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        main()

    except rospy.ROSInterruptException:
        pass
