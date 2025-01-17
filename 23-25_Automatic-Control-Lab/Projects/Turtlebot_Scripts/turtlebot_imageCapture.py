#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage

import numpy as np
import cv2 as cv
import os

directory = '/home/lw-lab/Pictures/Turtlebot/PlumeLess'
os.chdir(directory)


def imageCapture(CompressedImage):
    rate = rospy.Rate(2)
    image = np.asarray(bytearray(CompressedImage.data), dtype="uint8")

    imgDecode = cv.imdecode(image, cv.IMREAD_COLOR)
    # imgX = imgDecode.shape[:2]
    # cv.imshow('compressed image', imgDecode)
    # cv.waitKey(20)


    print()
    print("Files before saving image:", os.listdir(directory))

    cv.imwrite(f'{CompressedImage.header.seq}.jpg', imgDecode)

    print("Files after saving image:", os.listdir(directory))

    rate.sleep()

    # print(imgX)
    print('_____________')


if __name__ == "__main__":
    rospy.init_node('turtlebot3_ImageCapture')
    rospy.Subscriber("/camera/image/compressed", CompressedImage, imageCapture)
    rospy.spin()


