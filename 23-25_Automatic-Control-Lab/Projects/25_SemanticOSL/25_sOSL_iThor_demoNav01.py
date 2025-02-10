time = 0

def initRobot():
    # initiatialize the robot in a random position
    
    pass

def olfactionBranch(sourcePos, robotPos):
    # return odor concentration value based on the odor source and robot's position
    
    pass

def visionBranch():
    # get yolo object detection result
   
    # get the object positions, pass the info to world map

    pass

def worldMap(objectMetadata):
    # upon receiving the object metadata, update the world map
    
    pass

def sourceEstimation(worldMap):
    # estimate top odor source objects in world map using similarity
    
    pass


def fusion(target1, target2):
    # chart a path to the top odor source object
    
    # if odor concentration is decreasing, move to the second top odor source object
    
    # if there is obstacle on the way, avoid it
    pass

def obstacleAvoidance(target):
    # use a path planning algorithm to avoid obstacles and reach the odor source
    pass

def actionSelect(action):
    pass

def findSource():
    # if distance from the source object is less than a threshold, return True
    pass

def timeOut():
    pass

while True:
    time += 1
    initRobot()   # randomly generate robot position
    target1 = visionBranch()   # query the camera to get the vision target
    target2 = olfactionBranch()   # query the olfaction sensor to get the olfaction target
    target = fusion(target1, target2)    # fuse the visual and olfactory targets to get a final target
    action = obstacleAvoidance(target)   # 
    actionSelect(action)   # execute the command to move the robot position
    if findSource() or time > 200:   # check whether or not to stop the loop
        break