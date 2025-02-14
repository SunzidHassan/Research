import time
import math
import numpy as np
import pandas as pd
from ai2thor.controller import Controller
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util
from sOSL_plumeField import get_field_value  # Ensure this is available
import os

# ==========================
# SENSOR AND PERCEPTION
# ==========================

def olfactionBranch(sourcePos, controller, plumeConcentration, q_s=2000, D=10, U=0, tau=1000, del_t=1, psi_deg=0):
    """
    Computes odor concentration based on the odor source position and the robot's current position.
    
    :param sourcePos: Position(s) of the odor source(s)
    :param controller: The AI2-THOR Controller instance
    :param plumeConcentration: The previous plume concentration value
    :return: Updated plume concentration as an integer.
    """
    # Get robot's current position from the controller metadata
    robot_pos = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
    plumeConcentration = int(get_field_value(robot_pos, sources=sourcePos, q_s=q_s, D=D, U=U, tau=tau, del_t=del_t, psi_deg=psi_deg))
    return plumeConcentration

def objDetector(itemDF, controller, objDetectorModel):
    """
    Updates itemDF with YOLO object detection results and depth estimation.
    
    :param itemDF: Existing DataFrame of detected objects.
    :param controller: The AI2-THOR Controller instance.
    :param objDetectorModel: The YOLO model for object detection.
    :return: Updated DataFrame with new/updated detections.
    """
    itemList = []
    results = objDetectorModel(controller.last_event.frame)
    depthFrame = np.array(controller.last_event.depth_frame)
    
    for box in results[0].boxes:
        x, y, w, h = box.xywh[0]
        confidence = box.conf[0].item()
        class_name = objDetectorModel.names[int(box.cls[0].item())]

        # Estimate depth at the center of the bounding box
        x_rounded, y_rounded = round(x.item()), round(y.item())
        if 0 <= y_rounded < depthFrame.shape[0] and 0 <= x_rounded < depthFrame.shape[1]:
            depth_value = float(depthFrame[y_rounded, x_rounded])
        else:
            depth_value = np.nan

        itemList.append((class_name, confidence, (x.item(), y.item()), depth_value))
    
    new_df = pd.DataFrame(itemList, columns=["name", "conf", "vizLoc", "depth"])
    
    # Merge new detections with existing DataFrame
    for _, row in new_df.iterrows():
        existing_index = itemDF[itemDF["name"] == row["name"]].index
        if not existing_index.empty:
            # Update existing entry.
            itemDF.loc[existing_index, "conf"] = row["conf"]
            # Broadcast the vizLoc value as a Series with the same index.
            itemDF.loc[existing_index, "vizLoc"] = pd.Series([row["vizLoc"]] * len(existing_index), index=existing_index)
            itemDF.loc[existing_index, "depth"] = row["depth"]
        else:
            itemDF = pd.concat([itemDF, pd.DataFrame([row])], ignore_index=True)
            
    return itemDF


# ==========================
# LOCALIZATION
# ==========================

def globalLoc(itemDF, controller, image_width=300, image_height=300, fov_deg=90, camera_y_offset=0.0):
    """
    Updates the 'glb3DLoc' field in itemDF by converting image coordinates and depth
    into world coordinates based on the robot's position and yaw.
    
    :param itemDF: DataFrame containing object detections.
    :param controller: The AI2-THOR Controller instance.
    :param image_width: Image width in pixels.
    :param image_height: Image height in pixels.
    :param fov_deg: Camera field of view in degrees.
    :param camera_y_offset: Vertical offset for the camera relative to the robot.
    :return: Updated DataFrame with computed global coordinates.
    """
    robot_position = controller.last_event.metadata["agent"]["position"]
    robot_yaw = controller.last_event.metadata["agent"]["rotation"]["y"]  # in degrees

    for idx, row in itemDF.iterrows():
        current_global = row.get("glb3DLoc")
        # Check if current_global is None, or if it's a list/array with zero length,
        # or if it is a float and is NaN.
        if current_global is None or \
           (isinstance(current_global, (list, np.ndarray)) and len(current_global) == 0) or \
           (isinstance(current_global, float) and np.isnan(current_global)):
            pixel_coord = row.get("vizLoc")
            depth = row.get("depth")
            if pixel_coord is None or depth is None or pd.isna(depth):
                continue

            # Camera intrinsics
            cx = image_width / 2.0
            cy = image_height / 2.0
            f = (image_width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
            
            u, v = pixel_coord
            x_norm = (u - cx) / f
            y_norm = (v - cy) / f
            
            # Construct camera ray and scale by depth
            d = np.array([x_norm, y_norm, 1.0])
            norm_d = math.sqrt(x_norm**2 + y_norm**2 + 1)
            s = depth / norm_d
            p_cam = s * d
            
            # Rotate camera coordinates to world frame using robot yaw
            theta = math.radians(robot_yaw)
            R_yaw = np.array([
                [math.cos(theta), 0, math.sin(theta)],
                [0, 1, 0],
                [-math.sin(theta), 0, math.cos(theta)]
            ])
            p_rot = R_yaw.dot(p_cam)
            
            if isinstance(robot_position, dict):
                T_robot = np.array([robot_position["x"], robot_position["y"], robot_position["z"]])
            else:
                T_robot = np.array(robot_position)
            
            p_world_raw = p_rot + T_robot
            p_world = np.copy(p_world_raw)
            p_world[1] -= camera_y_offset
            
            itemDF.at[idx, "glb3DLoc"] = p_world.tolist()
    
    return itemDF


# ==========================
# REASONING
# ==========================

def goalSimilarity(itemDF, text_similarity_model, goal="food burning smell"):
    """
    Updates itemDF with a 'goalSimilarity' score by comparing the detected object names
    with a given goal description using cosine similarity.
    
    :param itemDF: DataFrame containing detected objects.
    :param text_similarity_model: SentenceTransformer model for generating embeddings.
    :param goal: The goal description to compare against.
    :return: Updated DataFrame with similarity scores.
    """
    goal_embedding = text_similarity_model.encode(goal, convert_to_tensor=True)
    
    for idx, row in itemDF.iterrows():
        if row.get("goalSimilarity") in [None, "", [], float('nan')] or pd.isna(row.get("goalSimilarity")):
            object_name = row.get("name")
            if object_name:
                name_embedding = text_similarity_model.encode(object_name, convert_to_tensor=True)
                similarity = util.cos_sim(goal_embedding, name_embedding).item()
                itemDF.at[idx, "goalSimilarity"] = similarity
                
    return itemDF

def fusion(itemDF, behavior, lastPlumeConcentration, plumeConcentration):
    """
    Performs fusion between vision and olfaction branches. Assigns search priority based on
    goal similarity and plume concentration change.
    
    :param itemDF: DataFrame containing detected objects.
    :param behavior: Current behavior mode (e.g., following odor source).
    :param lastPlumeConcentration: Plume concentration from the previous iteration.
    :param plumeConcentration: Current plume concentration.
    :return: Updated DataFrame with search priority modifications.
    """
    if not itemDF.empty:
        non_zero_priority_df = itemDF[itemDF["searchPriority"] != 0]
        if not non_zero_priority_df.empty:
            highest_similarity_idx = non_zero_priority_df["goalSimilarity"].idxmax()
            itemDF.at[highest_similarity_idx, "searchPriority"] = 1

    if behavior == 1:  # following odor source
        if lastPlumeConcentration > plumeConcentration:
            for idx, row in itemDF.iterrows():
                if row.get("searchPriority") == 1:
                    itemDF.at[idx, "searchPriority"] = 0

    return itemDF

# ==========================
# OBSTACLE AVOIDANCE & ACTION SELECTION (Placeholders)
# ==========================

def check_obstacle(controller, threshold=0.5):
    """
    Checks if there is an obstacle in a defined region of the depth frame.
    
    :param controller: The AI2-THOR Controller instance.
    :param threshold: Depth threshold (in meters) to consider as an obstacle.
    :return: True if an obstacle is detected, False otherwise.
    """
    depth_frame = np.array(controller.last_event.depth_frame)
    obstacle_region = depth_frame[150, 140:160]
    return np.any(obstacle_region < threshold)

def obstacleAvoidance(target):
    # Placeholder for obstacle avoidance logic using path planning.
    pass

def actionSelect(action):
    # Placeholder for action selection based on sensor input and target information.
    pass

def findSource():
    # Placeholder: Check whether the robot has reached the odor source.
    pass

# ==========================
# CONTROL LOOP
# ==========================

import os

def keyboard_control(controller, itemDF, yolo_model, text_similarity_model, save_path="itemDF.csv", max_time=150, goal="food burning smell"):
    """
    Processes keyboard input to move the robot and run the vision, localization, and reasoning branches.
    The loop will exit and save the final itemDF when the user presses 'q' or when max_time seconds elapse.
    """
    start_time = time.time()
    
    print("Keyboard control active. Use W/A/S/D for movement, Q to save & quit.")
    print(f"Program will auto-exit after {max_time} seconds.")
    
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time:
            print(f"\nTime limit of {max_time} seconds reached. Saving final itemDF to {save_path} and exiting.")
            _save_itemDF(itemDF, save_path)
            break
        
        key = input("Enter a move (W/A/S/D) or Q to quit: ").lower()
        
        if key == 'w':
            controller.step(action="MoveAhead")
            print("Moved Ahead.")
        elif key == 'a':
            controller.step(action="RotateLeft")
            print("Rotated Left.")
        elif key == 's':
            controller.step(action="MoveBack")
            print("Moved Back.")
        elif key == 'd':
            controller.step(action="RotateRight")
            print("Rotated Right.")
        elif key == 'q':
            print(f"Saving final itemDF to {save_path} and exiting.")
            _save_itemDF(itemDF, save_path)
            break
        else:
            print("Invalid input. Use W/A/S/D for movement, Q to quit.")
        
        # Vision branch: Update detections using YOLO
        itemDF = objDetector(itemDF, controller, yolo_model)
        # Global localization: Convert visual detections to world coordinates
        itemDF = globalLoc(itemDF, controller)
        # Reasoning: Compute goal similarity for detected objects
        itemDF = goalSimilarity(itemDF, text_similarity_model, goal=goal)
        
        # Print rounded values in terminal.
        print("Updated itemDF (rounded):")
        print(round_itemDF(itemDF))



def round_itemDF(df):
    """
    Returns a copy of the DataFrame where all float values (and those in lists/tuples)
    are rounded to 2 decimal places.
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
    """Helper function to create directory if needed, round values, and save itemDF."""
    import os
    parent_dir = os.path.dirname(save_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    # Round all float values to 2 decimal places before saving.
    rounded_df = round_itemDF(itemDF)
    rounded_df.to_csv(save_path, index=False)

# ==========================
# MAIN FUNCTION
# ==========================

def main():
    # Initialize an empty DataFrame for detected items.
    itemColumns = ["name", "conf", "vizLoc", "glb3DLoc", "goalSimilarity", "searchPriority"]
    itemDF = pd.DataFrame(columns=itemColumns)
    
    # Initialize models
    yolo_model = YOLO("models/YOLO/yolov8m.pt")
    text_similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize the AI2-THOR controller with desired configurations
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
    
    # Uncomment and integrate additional branches (e.g., olfaction, obstacle avoidance) as needed.
    # For now, we simply run the keyboard control loop that fuses vision, localization, and reasoning.
    keyboard_control(
        controller=controller,
        itemDF=itemDF,
        yolo_model=yolo_model,
        text_similarity_model=text_similarity_model,
        save_path="save/itemDF.csv",  # Update as needed.
        max_time=150,
        goal="food burning smell"
    )

if __name__ == "__main__":
    main()
