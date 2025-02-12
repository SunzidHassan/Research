import numpy as np
import pandas as pd
import math

from ai2thor.controller import Controller
from sOSL_plumeField import get_field_value
from sOSL_iThor_gndTruth import get_objects_centers
from sOSL_iThor_objDetector import obj_image_coordinate

from ultralytics import YOLO

from sentence_transformers import SentenceTransformer, util

def olfactionBranch(sourcePos):
    # return odor concentration value based on the odor source and robot's position
    lastPlumeConcentration = plumeConcentration
    robot_pos = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
    plumeConcentration = int(get_field_value(robot_pos, sources=sourcePos, q_s=2000, D=10, U=0, tau=1000, del_t=1, psi_deg=0))    
    return plumeConcentration



def visionBranch(itemDF, depthFrame):
    """
    Updates itemDF with YOLO object detection results and depth estimation.

    Parameters:
        itemDF (pd.DataFrame): Existing DataFrame of detected objects.
        depthFrame (np.ndarray): 2D depth map from the last event.

    Returns:
        pd.DataFrame: Updated DataFrame with new/updated detections.
    """
    # Get YOLO object detection results
    itemList = []
    results = model8m(controller.last_event.frame)

    for box in results[0].boxes:
        x, y, w, h = box.xywh[0]
        confidence = box.conf[0].item()
        class_name = model8m.names[int(box.cls[0].item())]

        # Estimate depth value at the object's center
        x_rounded, y_rounded = round(x.item()), round(y.item())

        if 0 <= y_rounded < depthFrame.shape[0] and 0 <= x_rounded < depthFrame.shape[1]:
            depth_value = float(depthFrame[y_rounded, x_rounded])
        else:
            depth_value = np.nan  # Assign NaN if out of bounds

        # Store detected object info
        itemList.append((class_name, confidence, (x.item(), y.item()), depth_value))

    # Convert itemList to DataFrame
    new_df = pd.DataFrame(itemList, columns=["name", "yoloConfidence", "[x,y] location", "depth"])

    # Merge new detections with existing itemDF
    for _, row in new_df.iterrows():
        existing_index = itemDF[itemDF["name"] == row["name"]].index

        if not existing_index.empty:
            # Update existing entry
            itemDF.loc[existing_index, "yoloConfidence"] = row["yoloConfidence"]
            itemDF.loc[existing_index, "[x,y] location"] = row["[x,y] location"]
            itemDF.loc[existing_index, "depth"] = row["depth"]
        else:
            # Append new object
            itemDF = pd.concat([itemDF, pd.DataFrame([row])], ignore_index=True)

    return itemDF

def globalLoc(itemDF,
              image_width=300,
              image_height=300,
              fov_deg=90,
              camera_y_offset=0.0  # Subtract this from the y-coordinate after transformation
             ):
    """
    For each object in itemDF that has an empty global location, compute its world coordinate 
    using the image coordinate and depth value. The function updates the 'globalLocation [x, y, z]'
    column in itemDF.
    
    Parameters:
      image_width: Image width in pixels (default: 300)
      image_height: Image height in pixels (default: 300)
      fov_deg: Camera field of view in degrees (default: 90)
      camera_y_offset: Vertical offset (in world units) to adjust the camera’s position relative to the robot
      
    Returns:
      Updated itemDF with computed global locations in "globalLocation [x, y, z]".
    """
    # Retrieve robot position and yaw from the controller metadata
    # (Assuming controller.last_event.metadata is available in your runtime environment.)
    robot_position = controller.last_event.metadata["agent"]["position"]
    robot_yaw = controller.last_event.metadata["agent"]["rotation"]["y"]  # in degrees

    # Loop through each row in itemDF to update missing global coordinates.
    for idx, row in itemDF.iterrows():
        # Check if the global location is empty or missing.
        current_global = row.get("globalLocation [x, y, z]")
        if current_global in [None, "", [], np.nan] or pd.isna(current_global):
            # Get the object's image coordinate and depth
            pixel_coord = row.get("[x,y] location")
            depth = row.get("depth")
            
            # Skip if we don't have a valid depth value or coordinate.
            if pixel_coord is None or depth is None or pd.isna(depth):
                continue
            
            # 1. Camera intrinsics
            cx = image_width / 2.0
            cy = image_height / 2.0
            # Compute focal length f (in pixel units)
            f = (image_width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
            
            # 2. Normalized image coordinates.
            # u,v are assumed to be provided in the "[x,y] location" column.
            u, v = pixel_coord
            x_norm = (u - cx) / f
            y_norm = (v - cy) / f
            
            # 3. Construct the ray in the camera frame.
            d = np.array([x_norm, y_norm, 1.0])
            norm_d = math.sqrt(x_norm**2 + y_norm**2 + 1)
            
            # 4. Scaling factor s so that s * ||d|| equals the measured depth.
            s = depth / norm_d
            
            # Compute the camera-frame coordinates.
            p_cam = s * d  # This gives [x_cam, y_cam, z_cam]
            
            # 5. Rotate the camera coordinates into the world frame using the robot's yaw.
            theta = math.radians(robot_yaw)
            R_yaw = np.array([
                [math.cos(theta), 0, math.sin(theta)],
                [0, 1, 0],
                [-math.sin(theta), 0, math.cos(theta)]
            ])
            p_rot = R_yaw.dot(p_cam)
            
            # 6. Convert robot_position to an array.
            if isinstance(robot_position, dict):
                T_robot = np.array([robot_position["x"], robot_position["y"], robot_position["z"]])
            else:
                T_robot = np.array(robot_position)
            
            # Add the translation (i.e., the robot's position).
            p_world_raw = p_rot + T_robot
            
            # 7. Adjust for the camera mounting offset in the y-coordinate.
            p_world = np.copy(p_world_raw)
            p_world[1] -= camera_y_offset
            
            # Update the DataFrame row with the computed global coordinate.
            # Here we store it as a list.
            itemDF.at[idx, "globalLocation [x, y, z]"] = p_world.tolist()
            
    return itemDF


def goalSimilarity(itemDF, goal="food burning smell"):
    """
    Updates the 'goalSimilarity' column in itemDF by computing the cosine similarity 
    between the given goal description and the object's name.
    
    Parameters:
        itemDF (pd.DataFrame): DataFrame containing detected objects with a 'name' column.
        goal (str): The target description to compare against (default: "food burning smell").
    
    Returns:
        pd.DataFrame: Updated DataFrame with computed goal similarity scores.
    """
    # Compute the embedding for the goal description
    goal_embedding = text_similarity_model.encode(goal, convert_to_tensor=True)

    for idx, row in itemDF.iterrows():
        # Check if goalSimilarity is empty or missing
        if row.get("goalSimilarity") in [None, "", [], float('nan')] or pd.isna(row.get("goalSimilarity")):
            object_name = row.get("name")
            
            if object_name:  # Ensure object name is valid
                # Compute the embedding for the object's name
                name_embedding = text_similarity_model.encode(object_name, convert_to_tensor=True)
                # Compute cosine similarity
                similarity = util.cos_sim(goal_embedding, name_embedding).item()
                
                # Update DataFrame
                itemDF.at[idx, "goalSimilarity"] = similarity

    return itemDF

def fusion(target1, target2):
    # chart a path to the top odor source object
    
    # if the robot is following odor source, and if odor concentration is decreasing, move to the second top odor source object
    if behavior == 1:
        if lastPlumeConcentration > plumeConcentration:
            behavior = 0
    
    # if there is obstacle on the way, avoid it
    pass


def check_obstacle(threshold=0.5):
    """
    Checks if there is an obstacle within depth[150, 140:160].
    
    Parameters:
        depth_frame (np.ndarray): 2D array of depth values.
        threshold (float): Depth threshold to consider an obstacle (default: 0.5m).
    
    Returns:
        bool: True if an obstacle is detected, False otherwise.
    """
    depth_frame = np.array(controller.last_event.depth_frame)
    
    # Extract the specified depth region
    obstacle_region = depth_frame[150, 140:160]
    
    # Check if any depth value in the region is less than the threshold
    obstacle = np.any(obstacle_region < threshold)
    
    return obstacle


def obstacleAvoidance(target):
    # use a path planning algorithm to avoid obstacles and reach the odor source
    if check_obstacle():
        behavior = 0
        pass
    

def actionSelect(action):
    pass


def findSource():
    # if distance from the source object is less than a threshold, return True
    pass


def timeOut():
    pass


if __name__ == "__main__":
    time = 0
    behavior = 1
    lastPlumeConcentration = 0
    plumeConcentration = 0

    itemColumns = ["name", "yoloConf", "vizLoc", "glb3DLoc", "goalSimilarity"]

    # Create an empty DataFrame
    itemDF = pd.DataFrame(columns=itemColumns)

    model8m = YOLO("models/YOLO/yolov8m.pt")
    
    # Load the sentence transformer model for text similarity
    text_similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # initiatialize the robot in a random position
    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,

        scene="FloorPlan1",
        # kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
        # living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
        # bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
        # bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

        # step sizes
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,

        # image modalities
        renderDepthImage=True,
        renderInstanceSegmentation=True,

        # camera properties
        width=300,
        height=300,
        fieldOfView=90
    )


    objects = controller.last_event.metadata["objects"]
    gt_target_items = ["Microwave"]
    
    gt_object_centers = get_objects_centers(objects, gt_target_items)
    
    allObjects = controller.last_event.metadata["objects"]
    target_items = ["Toaster"]

    visionResults = visionBranch()
    

    while True:
        time += 1
        target1 = visionBranch()   # query the camera to get the vision target
        target2 = olfactionBranch()   # query the olfaction sensor to get the olfaction target
        target = fusion(target1, target2)    # fuse the visual and olfactory targets to get a final target
        action = obstacleAvoidance(target)   # 
        actionSelect(action)   # execute the command to move the robot position
        if findSource() or time > 200:   # check whether or not to stop the loop
            break
        
        

# vision > 3d world map > target list > search and update list until localization