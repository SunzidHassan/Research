import time
import math
import numpy as np
import pandas as pd
import yaml
import os
import base64
import requests
import io
from PIL import Image
import cv2
import re

from ai2thor.controller import Controller
from ultralytics import YOLO
# from openai import OpenAI  # if needed

import open3d as o3d

    
# ==========================
# HELPER FUNCTION: Compress Image
# ==========================
def compress_image(image_array, size=(64, 64)):
    """
    Resizes the input image array to the specified size and returns the compressed image bytes.
    """
    im = Image.fromarray(image_array.astype('uint8'))
    im = im.resize(size)
    buffer = io.BytesIO()
    im.save(buffer, format="JPEG")
    return buffer.getvalue()

# Get object centers
def get_objects_centers(objects, target_names):
    """
    Filters a list of objects for those whose name contains any of the target names
    and extracts their center x and y coordinates as a NumPy array.
    
    The center is taken from the object's 'axisAlignedBoundingBox' field if available,
    otherwise from the object's 'position'.
    
    Parameters:
        objects (list): A list of dictionaries representing scene objects.
        target_names (list): A list of strings representing substrings to match in the object name.
                             For example: ["Apple", "Bread"]
        
    Returns:
        np.ndarray: A tensor of shape (n, 2) where each row contains the [x, y]
                    coordinates of a matching object.
    """
    centers = []
    
    for obj in objects:
        name = obj.get("name", "")
        # Check if any of the target substrings is in the object's name
        if any(target in name for target in target_names):
            # Prefer the center from axisAlignedBoundingBox if available; otherwise, use the object's position.
            center = obj.get("axisAlignedBoundingBox", {}).get("center", obj.get("position"))
            
            if center is not None and "x" in center and "y" in center and "z" in center:
                centers.append([center["x"], center["y"], center["z"]])
            else:
                print(f"Center coordinates not available for object: {name}")
                
    return np.array(centers)


def get_distance_to_source(controller, sourcePos):
    """
    Given the source position (a 3D coordinate [x, y, z]),
    obtains the robot's current position from controller metadata,
    and returns the Euclidean distance on the ground plane (using x and z).
    """
    agent_pos = controller.last_event.metadata["agent"]["position"]
    robot_pos = np.array([agent_pos["x"], agent_pos["z"]])
    source_pos = np.array([sourcePos[0], sourcePos[2]])
    return np.linalg.norm(robot_pos - source_pos)

# ==========================
# SENSOR FUNCTIONS
# ==========================

def get_field_value(x, z, sources, q_s=2000, D=1000, U=0, tau=1000, del_t=1, psi_deg=0):
    """
    Computes the odor field value at a single (x, z) coordinate as the sum of contributions
    from one or more odor sources.

    Parameters:
        x, z       (float): Coordinates at which to evaluate the field.
        sources    (ndarray or list): A collection of source positions, where each source is [x_s, y_s, z_s].
        q_s        (float): Source strength.
        D          (float): Diffusion coefficient.
        U          (float): Advection velocity (set to 0 if no airflow).
        tau        (float): Time or scaling parameter.
        del_t      (float): Time step.
        psi_deg    (float): Angle in degrees for rotation (direction of advection; irrelevant if U==0).

    Returns:
        (float): The computed field value at the coordinate (x, z) as the sum of contributions
                 from all sources.
    """
    # Convert psi from degrees to radians
    psi = math.radians(psi_deg)
    
    # Compute lambda; note that if U==0, lambda simplifies to sqrt(D*tau)
    lambd = math.sqrt((D * tau) / (1 + (tau * U**2) / (4 * D)))
    
    total = 0.0
    # Loop over each source
    for source in sources:
        x_s, y_s, z_s = source  # Unpack the source coordinates; ignore y_s here.
        
        # Compute differences in x and z relative to the odor source
        delta_x = x - x_s
        delta_z = z - z_s
        
        # Euclidean distance in the X-Z plane
        r = math.sqrt(delta_x**2 + delta_z**2)
        
        # Avoid division by zero if r==0
        if r == 0:
            contribution = 0
        else:
            # Compute the rotated z coordinate (this incorporates advection if U != 0)
            rotated_z = -delta_x * math.sin(psi) + delta_z * math.cos(psi)
            
            contribution = (q_s / (4 * math.pi * D * r)) * math.exp((-rotated_z * U) / (2 * D) - (r / lambd) * del_t)
        
        total += contribution
        
    return total


def olfactionBranch(sourcePos, controller,
                    q_s=200, D=10, U=0, tau=10, del_t=10, psi_deg=0):
    """
    Computes odor concentration based on the odor source position and the robot's current position.
    """
    robot_x, robot_y, robot_z = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
    plumeConcentration = round(get_field_value(robot_x, robot_z, sourcePos, q_s=q_s, D=D, U=U, tau=tau, del_t=del_t, psi_deg=psi_deg),4)
    return plumeConcentration

def boxDepth(x, y, w, h, controller):
    vMin = y - h//2
    vMax = y + h//2
    hMin = x - w//2
    hMax = x + w//2
    depthFrame = controller.last_event.depth_frame
    boxDepth = np.percentile(depthFrame[vMin:vMax, hMin:hMax], 90)
    return round(boxDepth, 1)


def visionBranch(itemDF, controller, visionBranchModel, confThr = 0.5):
    """
    Updates itemDF with YOLO object detection results and depth estimation.
    """
    itemList = []
    results = visionBranchModel(controller.last_event.frame)
    depthFrame = np.array(controller.last_event.depth_frame)
    
    for box in results[0].boxes:
        x, y, w, h = box.xywh[0]
        confidence = box.conf[0].item()
        class_name = visionBranchModel.names[int(box.cls[0].item())]

        # Estimate depth at the center of the bounding box.
        x, y, w, h = round(x.item()), round(y.item()), round(w.item()), round(h.item())
        if 0 <= y < depthFrame.shape[0] and 0 <= x < depthFrame.shape[1]:
            depth_value = boxDepth(x, y, w, h, controller)
        else:
            depth_value = np.nan

        if confidence > confThr:
            itemList.append((class_name, confidence, (x.item(), y.item()), depth_value))
    
    new_df = pd.DataFrame(itemList, columns=["name", "conf", "vizLoc", "depth"])
    
    for _, row in new_df.iterrows():
        existing_index = itemDF[itemDF["name"] == row["name"]].index
        if not existing_index.empty:
            itemDF.loc[existing_index, "conf"] = row["conf"]
            itemDF.loc[existing_index, "vizLoc"] = pd.Series([row["vizLoc"]] * len(existing_index), index=existing_index)
            itemDF.loc[existing_index, "depth"] = row["depth"]
        else:
            itemDF = pd.concat([itemDF, pd.DataFrame([row])], ignore_index=True)
            
    return itemDF

def img_to_points(controller, width: int = 300, height: int = 300):
    depth_img = np.array(controller.last_event.depth_frame)
    depth_img = np.clip(depth_img, 0, 3.3)  # Clip depth values to 10 meters
    depth_scale = 1000.0

    width, height = depth_img.shape[1], depth_img.shape[0]
    fx, fy = 150, 150  # Focal lengths
    cx, cy = width / 2, height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    color_image = o3d.geometry.Image(np.array(controller.last_event.frame))
    depth_image = o3d.geometry.Image((depth_img / depth_scale).astype(np.float32)) #300X300 image

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_image,
        depth = depth_image,
        convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)

    pts = np.asarray(pcd.points)
    
    return pts



# ==========================
# Fusion Branch
# ==========================

# ==========================
# AUTOMATIC CONTROL LOOP
# ==========================

def goal_direction(controller, target_items):
    objects = controller.last_event.metadata["objects"]
    sourcePos = get_objects_centers(objects, target_items)
    target = sourcePos[0]  
    target_x, target_y, target_z = target
    
    robotPos = controller.last_event.metadata["agent"]["position"]
    robotYaw = controller.last_event.metadata["agent"]["rotation"]["y"]

    # Calculate differences in x and z (ignoring y)
    dx = target_x - robotPos["x"]
    dz = target_z - robotPos["z"]

    # Compute the angle (in radians) from the robot to the target
    # Assuming forward is along positive z-axis, use: angle = atan2(delta_x, delta_z)
    target_angle_rad = math.atan2(dx, dz)
    target_angle_deg = math.degrees(target_angle_rad)

    # Compute the required rotation relative to the robot's current yaw
    angle_to_rotate = target_angle_deg - robotYaw

    # Normalize the angle to be between -180 and 180 degrees
    angle_to_rotate = np.deg2rad((angle_to_rotate + 180) % 360 - 180)
    
    return angle_to_rotate


def gap_angle(controller):
    pts = img_to_points(controller)
    
    # Assume pts is an (N, 3) array from your point cloud,
    # and that a lower y value corresponds to points from the lower half of the image.
    # Choose an appropriate threshold for y. For instance, you might use the median value.
    pts_lower = pts[pts[:, 1] > np.median(pts[:, 1])]


    # Compute polar coordinates (r, theta) for each point in the horizontal plane.
    # Assuming pts[:,0] is x (horizontal) and pts[:,2] is depth (forward).
    angles = np.arctan2(pts_lower[:, 0], pts_lower[:, 2])  # 300x300 values in radians

    # Compute horizontal distances using the x and z coordinates.
    distances = np.linalg.norm(pts_lower[:, [0, 2]], axis=1)

    # Create a histogram of angles (simulate a 1D laser scan)
    # Dividing the scene into 90 bins
    # Weights are inversely proportional to the distance, so closer obstacles contribute more.
    num_bins = 90
    hist, bin_edges = np.histogram(angles, bins=num_bins, weights=1/(distances + 1e-6))

    # Find the bin with the smallest weighted obstacle presence
    gap_index = np.argmin(hist)
    gap_angle = (bin_edges[gap_index] + bin_edges[gap_index + 1]) / 2  # Middle angle of the bin
    
    return gap_angle


def fusion_control(controller, itemDF, yolo_model, source_position, 
                 save_path="itemDF.csv", step_threshold = 50, max_time=150, goal="", 
                 dist_threshold=1.0, stepMagnitude=0.5):
    """
    Automatic control loop.
    Each iteration:
      1. Vision branch provides environment knowledge.
      2. Builds the action table (with obstacles) containing only Action and Obstacle.
      3. Extracts scene objects and computes target object centers.
      4. Computes the minimum distance from the robot to any target object.
         If the distance is below dist_threshold, the loop terminates.
      5. Calls GPT to select the best action based solely on the attached (compressed) image.
         GPT will decide the action by analyzing the image.
      6. Logs the time, robot position (x, z), robot yaw, and GPT decision.
    """
    
    step_count = 1    
    start_time = time.time()
    logDF = pd.DataFrame(columns=["step", "robot_x", "robot_z", "robot_yaw", 
                                  "target_object", "goal_heading", "obstacle_heading", "final_heading", "concentration"])
    
    print("Fusion control active. Executing actions until timeout or target reached.")
    
    while True:
        print("\n=============================")
        print("New Step")
        print("=============================\n")
        # elapsed_time = time.time() - start_time
        print(f"Steps: {step_count}/{step_threshold}")
        
        # Ground truth distance to source position
        if source_position.size > 0:
            distances = [get_distance_to_source(controller, center) for center in source_position]
            min_distance = min(distances)
        else:
            min_distance = float('inf')
        
        print(f"Current minimum distance to target: {min_distance:.2f}")
        
        # get current odor concentration
        currentConcentration = olfactionBranch(source_position, controller)
        
        # # Variable obstacle threshold
        # obstacle_threshold = np.interp(currentConcentration, [0, 0.3], [1, 0.6])
         
        # Retrieve robot's current pose.
        agent_meta = controller.last_event.metadata["agent"]
        robot_x = agent_meta["position"].get("x", None)
        robot_z = agent_meta["position"].get("z", None)  # using z for ground plane coordinate
        robot_yaw = agent_meta["rotation"].get("y", None)
        
        
        # Log the current step.
        log_entry = {
            "step": step_count,
            "robot x": robot_x,
            "robot z": robot_z,
            "robot yaw": robot_yaw,
            "target_object": None,
            "goal_heading": None,
            "obstacle_heading": None,
            "final_heading": None,
            "concentration": None
        }
        logDF = pd.concat([logDF, pd.DataFrame([log_entry], columns=logDF.columns)], 
                          ignore_index=True)
        
        # Implement variable step magnitude based on current concentration
        stepMagnitude = np.interp(currentConcentration, [0, 0.3], [0.7, 0.25])

 
        # Ensure goal_direction is defined. For example, assume straight ahead:
        # target_items = ["HousePlant"]
        target_items = ["Microwave"]
        goalAng = goal_direction(controller, target_items)
        print(f"Goal direction: {goalAng:.2f}")
        # calculate gaps
        gapAng = gap_angle(controller)
        print(f"Gap direction: {gapAng:.2f}")

        # Blend the gap direction with the goal direction (tuning parameter)
        alpha = 0.5
        steering_angle = np.rad2deg(alpha * goalAng + (1 - alpha) * gapAng)
        steering_angle = (steering_angle + 180) % 360 - 180
        
        print(f"Steering angle: {steering_angle:.2f}")
        
        # Map action id to a controller action.
        if steering_angle < 0:
            controller.step(action="RotateLeft", degrees=abs(steering_angle))
            controller.step(action="MoveAhead", moveMagnitude=stepMagnitude)
            print("Executing action: Rotate Left.")
        else:
            controller.step(action="RotateRight", degrees=steering_angle)
            controller.step(action="MoveAhead", moveMagnitude=stepMagnitude)
            print("Executing action: Rotate Right.")
        
        # Save the current vision frame.
        # frame_filename = f"save/{step_count}.png"
        # cv2.imwrite(frame_filename, controller.last_event.cv2img)
        # print(f"Saved vision frame as {frame_filename}")
        
        # print(f"Robot x: {robot_x}, Robot z: {robot_z}")
        # cv2.imshow("AI2-THOR", controller.last_event.cv2img)
        # cv2.waitKey(int(1000))
        
        # Check termination condition AFTER logging the decision.
        if min_distance < dist_threshold:
            print(f"Robot is within {dist_threshold} of the target. Mission accomplished!")
            logDF.to_csv("save/trajectory_log.csv", index=False)
            break
        
        # Check step limit.
        if step_count >= 40:
            print(f"Step limit of 40 reached. Saving log and exiting.")
            logDF.to_csv("save/trajectory_log.csv", index=False)
            break
        
        step_count += 1
        
        time.sleep(1)


def round_itemDF(df):
    """
    Returns a copy of the DataFrame with all float values (and those in lists/tuples)
    rounded to 2 decimal places.
    """
    def round_cell(x):
        if isinstance(x, float):
            return round(x, 2)
        elif isinstance(x, (list, tuple)):
            return [round(i, 2) if isinstance(i, float) else i for i in x]
        else:
            return x
    return df.applymap(round_cell)

def _save_itemDF(itemDF, save_path):
    """Creates directory if needed, rounds values, and saves itemDF as CSV."""
    parent_dir = os.path.dirname(save_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    rounded_df = round_itemDF(itemDF)
    rounded_df.to_csv(save_path, index=False)

# ==========================
# MAIN FUNCTION
# ==========================
def main():
    stepMagnitude = 0.25
    
    
    # config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    # api_key = config['OPENAI_KEY']
    # gpt_model = config['OPENAI_CHAT_MODEL']
    
    itemColumns = ["name", "conf", "vizLoc", "glb3DLoc", "goalSimilarity", "searchPriority"]
    itemDF = pd.DataFrame(columns=itemColumns)
    
    yolo_model = YOLO("models/YOLO/yolov8s.pt")
    
    cv2.namedWindow("AI2-THOR", cv2.WINDOW_NORMAL)
    
    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene="FloorPlan1",
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width=300,
        height=300,
        fieldOfView=90
    )
    
    # goal = "smoke"
    # target_items = ["Microwave"]
    
    goal = "tree smell"
    target_items = ["HousePlant"]
    
    objects = controller.last_event.metadata["objects"]
    sourcePos = get_objects_centers(objects, target_items)

    # Obtain current scene objects.
    if target_items == ['Microwave']:
        x, y, z = sourcePos[0]
        z += 0.5
        sourcePos = np.array([[x, y, z]])
    elif target_items == ['GarbageCan']:
        x, y, z = sourcePos[0]
        x += 0.25
        sourcePos = np.array([[x, y, z]])
        

    # # Microwave Starting position 1
    # controller.step(
    #     action="Teleport",
    #     position=dict(x=1.5, y=0.9, z=1.5),
    #     rotation=dict(x=0, y=180, z=0)
    # )

    # controller.step(
    #     "MoveAhead",
    #     moveMagnitude=0.01
    # )
    
    # # Microwave Starting position 2
    # controller.step(
    #     action="Teleport",
    #     position=dict(x=2, y=0.9, z=0),
    #     rotation=dict(x=0, y=0, z=0)
    # )

    # controller.step(
    #     "MoveAhead",
    #     moveMagnitude=0.01
    # )
    
    # # Microwave Starting position 3
    # controller.step(
    #     action="Teleport",
    #     position=dict(x=-1, y=0.9, z=2),
    #     rotation=dict(x=0, y=0, z=0)
    # )

    # controller.step(
    #     "MoveAhead",
    #     moveMagnitude=0.01
    # )
    
    
    # # Garbage Start Pos 1: facing back to the garbage bin
    # controller.step(
    #     action="Teleport",
    #     position=dict(x=1.5, y=0.9, z=2),
    #     rotation=dict(x=0, y=90, z=0),
    # )

    # controller.step(
    #     "MoveAhead",
    #     moveMagnitude=0.01
    # )

    # # Garbage Start Pos 2: upper left corner
    # controller.step(
    #     action="Teleport",
    #     position=dict(x=2, y=0.9, z=-1.5),
    #     rotation=dict(x=0, y=180, z=0),
    # )

    # controller.step(
    #     "MoveAhead",
    #     moveMagnitude=0.01
    # )


    # # Garbage Start Pos 3:
    # controller.step(
    #     action="Teleport",
    #     position=dict(x=-1, y=0.9, z=-1.5),
    #     rotation=dict(x=0, y=90, z=0),
    # )

    # controller.step(
    #     "MoveAhead",
    #     moveMagnitude=0.01
    # )

    fusion_control(
        controller=controller,
        itemDF=itemDF,
        yolo_model=yolo_model,
        # api_key=api_key,
        # gpt_model=gpt_model,
        source_position=sourcePos,
        save_path="save/itemDF.csv",
        max_time=200,
        goal=goal,
        dist_threshold=0.8,
        stepMagnitude=stepMagnitude
    )

if __name__ == "__main__":
    main()
