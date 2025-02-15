import pandas as pd
import numpy as np
import math

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
