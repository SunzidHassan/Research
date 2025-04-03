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
import networkx as nx

from ai2thor.controller import Controller
from ultralytics import YOLO
# from openai import OpenAI  # if needed
from sentence_transformers import SentenceTransformer, util

import open3d as o3d
from scipy.ndimage import gaussian_filter

    
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

def extract_object_table(controller):
    """
    Extracts a table of object types, confidence, and rounded positions
    from the provided metadata.

    Parameters:
        metadata (list): A list of dictionaries, each representing object metadata.

    Returns:
        list: A list of dictionaries, each with keys 'objectType', 'Conf', and 'Position'.
    """
    metadata = controller.last_event.metadata["objects"]
    result = []
    CONFIDENCE = 1
    for obj in metadata:
        obj_type = obj.get("objectType", "N/A")
        pos = obj.get("position", {})
        # Round each coordinate to 2 decimal places
        x = round(pos.get("x", 0), 2)
        y = round(pos.get("y", 0), 2)
        z = round(pos.get("z", 0), 2)
        position_str = f"{x}, {y}, {z}"
        result.append({
            "objectType": obj_type,
            "Conf": CONFIDENCE,
            "Position": position_str
        })
    return result

def parse_position_string(pos_str):
    # Expecting a string like "x, y, z"
    return np.array([float(val.strip()) for val in pos_str.split(',')])

# ==========================
# OLFACTION FUNCTIONS
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


# ==========================
# VISION FUNCTIONS
# ==========================

def boxDepth(x, y, w, h, controller):
    vMin = y - h//2
    vMax = y + h//2
    hMin = x - w//2
    hMax = x + w//2
    depthFrame = controller.last_event.depth_frame
    boxDepth = np.percentile(depthFrame[vMin:vMax, hMin:hMax], 90)
    return round(boxDepth, 1)


def rotate_point(x_cam, z_cam, yaw_deg):
    # Convert yaw angle to radians.
    yaw_rad = np.deg2rad(yaw_deg)
    # Build the rotation matrix for yaw.
    R = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)],
                  [np.sin(yaw_rad),  np.cos(yaw_rad)]])
    rotated = R.dot(np.array([x_cam, z_cam]))
    return rotated[0], rotated[1]


def coord23D(x, y, w, h, controller):
    image_width = 300  # controller.last_event.frame.shape[1]
    image_height = 300 # controller.last_event.frame.shape[0]
    h_fov = v_fov = 90
    
    center_u = image_width / 2.0  # 150
    center_v = image_height / 2.0  # 150

    # Angular resolution: degrees per pixel.
    angle_per_pixel_h = (h_fov / 2.0) / center_u  # 0.3 degrees per pixel
    angle_per_pixel_v = (v_fov / 2.0) / center_v    # 0.3 degrees per pixel

    # Angular offsets from the center (in degrees).
    theta_h_deg = (x - center_u) * angle_per_pixel_h
    theta_v_deg = (y - center_v) * angle_per_pixel_v

    # Convert angles to radians.
    theta_h = np.deg2rad(theta_h_deg)
    theta_v = np.deg2rad(theta_v_deg)

    # Compute the depth of the object.
    d = boxDepth(x, y, w, h, controller)
    
    # Compute 3D coordinates in the camera frame.
    x_cam = d * np.sin(theta_h) * np.cos(theta_v)
    y_cam = d * np.sin(theta_v)  # Flip sign if image v increases downward
    z_cam = d * np.cos(theta_h) * np.cos(theta_v)

    # Combine into camera-space vector
    p_cam = np.array([x_cam, y_cam, z_cam])

    # Step 2: Get robot's yaw and compute rotation matrix
    robot_yaw_deg = controller.last_event.metadata["agent"]["rotation"]["y"]
    theta = math.radians(robot_yaw_deg)

    R_yaw = np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ])

    # Rotate to align with global axes
    p_rot = R_yaw @ p_cam

    # Step 3: Translate using agent's position
    agent_pos = controller.last_event.metadata["agent"]["position"]
    x_global = round(agent_pos["x"] + p_rot[0], 2)
    y_global = round(agent_pos["y"] + p_rot[1], 2)
    z_global = round(agent_pos["z"] + p_rot[2], 2)

    # Step 4: Return global coordinates
    return x_global, y_global, z_global



def visionBranch(model, itemDF, controller, confThr=0.3):
    """
    Updates itemDF with YOLO object detection results and depth estimation.
    
    For each detection:
      - Compute the 3D coordinate (via coord23D).
      - If an object with the same type exists in itemDF, check its position.
        If the Euclidean distance is less than 0.5, update that entry by averaging the positions.
      - Otherwise, append a new row.
    """
   
    results = model(np.array(controller.last_event.frame))
    
    for box in results[0].boxes:
        # Get xywh coordinates and confidence from the detection.
        x, y, w, h = box.xywh[0]
        confidence = box.conf[0].item()  # Convert confidence from tensor to float.
        if confidence > confThr:
            # Get the actual object name using the model's names dictionary.
            className = model.names[int(box.cls[0].item())]
            x, y, w, h = round(x.item()), round(y.item()), round(w.item()), round(h.item())
            
            # Compute the global 3D coordinate for this detection.
            x_cam, y_cam, z_cam = coord23D(x, y, w, h, controller)
            new_position = np.array([x_cam, y_cam, z_cam])
            
            updated = False
            # Check if an entry with the same object type already exists in itemDF.
            for idx, row in itemDF.iterrows():
                if row['objectType'] == className:
                    # Convert the stored string "x, y, z" to a numpy array.
                    existing_position = np.array([float(val.strip()) for val in row['Position'].split(',')])
                    # Compute the Euclidean distance.
                    dist = np.linalg.norm(new_position - existing_position)
                    if dist < 0.5:
                        # Average the two positions.
                        avg_position = (new_position + existing_position) / 2.0
                        itemDF.at[idx, 'Position'] = f"{avg_position[0]}, {avg_position[1]}, {avg_position[2]}"
                        updated = True
                        break
            # If no similar object exists, append a new entry.
            if not updated:
                new_row = pd.DataFrame({
                    "objectType": [className],
                    "Conf": [confidence],
                    "Position": [f"{x_cam}, {y_cam}, {z_cam}"]
                })
                itemDF = pd.concat([itemDF, new_row], ignore_index=True)
                
    return itemDF


def initialize_envKnowledge(controller, model, itemDF, confThr=0.3):
    """
    Initializes the itemDF table with YOLO object detection results.
    """
    for i in range(4):
        itemDF = visionBranch(model, itemDF, controller)
        itemDF_list = itemDF.to_dict(orient='records')
        itemDF = pd.DataFrame(itemDF_list)
        print(itemDF)
        print("\n")
    
        controller.step(
            "RotateLeft")
    itemDF = add_goal_similarity(itemDF, "burning smell")
    return itemDF

def generate_vision_map(df, grid_size=50, sigma=1):
    """
    Generate a vision probability map from a DataFrame of object detections.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns "objectType", "Conf", "Position", and "goalSim".
        The "Position" column is expected to be a string in the form "x_cam, y_cam, z_cam".
    grid_size : int, optional
        The size of the grid (default is 50, for a 50x50 grid).
    sigma : float, optional
        Standard deviation for Gaussian smoothing (default is 1).

    Returns
    -------
    vision_map : numpy.ndarray
        A normalized 2D array of shape (grid_size, grid_size) representing the vision probability map.
    """
    vision_map = np.zeros((grid_size, grid_size))

    # Loop through each row in the DataFrame to accumulate probabilities.
    for _, row in df.iterrows():
        # Parse the position string "x_cam, y_cam, z_cam"
        pos_str = row["Position"]
        parts = pos_str.split(',')
        x = int(float(parts[0].strip()))
        y = int(float(parts[1].strip()))
        
        # Compute a combined probability weight from confidence and goalSim.
        p = row["Conf"] * row["goalSim"]
        
        # Add the weight to the corresponding grid cell.
        # Here, we assume that the grid uses array indexing with y as the row and x as the column.
        if 0 <= x < grid_size and 0 <= y < grid_size:
            vision_map[y, x] += p

    # Apply Gaussian smoothing to spread the probability locally.
    vision_map = gaussian_filter(vision_map, sigma=sigma)
    
    # Normalize the map so that the probabilities sum to 1.
    vision_map = vision_map / np.sum(vision_map)
    
    return vision_map
# ==========================
# CONTROL LOOP
# ==========================

def create_graph_from_positions(positions, threshold=0.3):
    """
    Creates an undirected graph from a list of position dictionaries.
    Each node corresponds to a position and an edge is added if the
    Euclidean distance between two positions is less than or equal to threshold.

    Args:
        positions (list): List of dicts with keys 'x', 'y', 'z'
        threshold (float): Maximum distance to consider two positions connected

    Returns:
        networkx.Graph: A graph where nodes have an attribute 'pos' containing (x, y, z)
    """
    G = nx.Graph()
    
    # Add nodes with position attributes.
    for i, pos in enumerate(positions):
        G.add_node(i, pos=(pos['x'], pos['y'], pos['z']))
    
    # Add edges between nodes if the distance is within the threshold.
    for i in range(len(positions)):
        p1 = np.array([positions[i]['x'], positions[i]['y'], positions[i]['z']])
        for j in range(i + 1, len(positions)):
            p2 = np.array([positions[j]['x'], positions[j]['y'], positions[j]['z']])
            dist = np.linalg.norm(p1 - p2)
            if dist <= threshold:
                # You can store the distance as a weight.
                G.add_edge(i, j, weight=dist)
    return G

def find_nearest_node(graph, position):
    """
    Finds the nearest node in the graph to a given 3D position.

    Args:
        graph (networkx.Graph): Graph with nodes having a 'pos' attribute (x, y, z).
        position (tuple, list, or dict): The (x, y, z) position to compare.

    Returns:
        tuple: (nearest_node, distance) where nearest_node is the node index and distance is the Euclidean distance.
    """
    # Convert position to a numpy array if it's a dict.
    if isinstance(position, dict):
        position = np.array([position['x'], position['y'], position['z']])
    else:
        position = np.array(position)
    
    min_dist = float('inf')
    nearest_node = None
    for node in graph.nodes():
        node_pos = np.array(graph.nodes[node]['pos'])
        dist = np.linalg.norm(node_pos - position)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node, min_dist

def add_goal_similarity(table, goal_phrase):
    """
    Updates a DataFrame with a new 'goalSim' column and sorts by (Conf * goalSim).
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    goal_embedding = model.encode(goal_phrase, convert_to_tensor=True)

    for idx, row in table.iterrows():
        object_embedding = model.encode(row["objectType"], convert_to_tensor=True)
        cosine_sim = util.pytorch_cos_sim(object_embedding, goal_embedding).item()
        table.at[idx, "goalSim"] = cosine_sim

    table["Conf_goalSim"] = table["Conf"] * table["goalSim"]
    table = table.sort_values(by="Conf_goalSim", ascending=False).drop(columns="Conf_goalSim")
    return table



def fusion_control(controller, itemDF, yolo_model, source_position, 
                 save_path="itemDF.csv", step_threshold = 50, max_time=150, goal_phrase="", 
                 dist_threshold=1.0, stepMagnitude=0.5):
    """
    Automatic control loop.
    Each iteration:
    - Vision branch provides environment knowledge.
        - List of objects, detection confidence, 3D location and goal similarity.
    - Olfaction branch provides odor concentration.
    - Fusion branch combines vision and olfaction data.
        - Approach the object with highest goal similarity.
        - If odor concentration decreases while distance to the object decreaes,
        discard the object and approach the object with second highest goal similarity.
        - Otherwise, if the robot reaches within a threshold distance to the object,
        terminate the loop.
    - Logs the time, robot position (x, z), robot yaw...
    """
    
    step_count = 1    
    start_time = time.time()
    logDF = pd.DataFrame(columns=["step", "robot_x", "robot_z", "robot_yaw", 
                                  "target_object", "concentration"])
    
    print("Fusion control active. Executing actions until timeout or target reached.")
    
    # ========================== #
    ## Vision Branch: environment knowledge -> coordiante
    # Get environment knowledge
    envKnowledge = initialize_envKnowledge(controller, yolo_model, itemDF, confThr=0.5)
    navKnowledge = add_goal_similarity(envKnowledge, goal_phrase)
    
    # Get reachable positions
    positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    # Generate a graph of rechable positions
    graph = create_graph_from_positions(positions, threshold=0.3)

    while True:
        print("\n=============================")
        print("New Step")
        print("=============================\n")
        # elapsed_time = time.time() - start_time
        print(f"Steps: {step_count}/{step_threshold}")


        # ========================== #
        ## Ground truth measurements
        
        # Ground truth distance to source position
        if source_position.size > 0:
            distances = [get_distance_to_source(controller, center) for center in source_position]
            min_distance = min(distances)
        else:
            min_distance = float('inf')

        print(f"Current minimum distance to target: {min_distance:.2f}")

        # Check termination condition AFTER logging the decision.
        if min_distance < dist_threshold:
            print(f"Robot is within {dist_threshold} of the target. Mission accomplished!")
            logDF.to_csv("save/trajectory_log.csv", index=False)
            break
        
        # Check step limit.
        if step_count >= step_threshold:
            print(f"Step limit of {step_threshold} reached. Saving log and exiting.")
            logDF.to_csv("save/trajectory_log.csv", index=False)
            break
        
        # Retrieve robot's current pose.
        agent_meta = controller.last_event.metadata["agent"]
        robot_x = agent_meta["position"].get("x", None)
        robot_z = agent_meta["position"].get("z", None)  # using z for ground plane coordinate
        robot_yaw = agent_meta["rotation"].get("y", None)
        
        # ========================== #
        ## Olfaction Branch: robot coordinate -> odor concentration

        # get current odor concentration
        current_odor_concentration = olfactionBranch(source_position, controller)
        
        prev_odor_concentration = getattr(fusion_control, "prev_odor_concentration", None)
        
        # If no previous value exists, initialize it.
        if prev_odor_concentration is None:
           prev_odor_concentration = current_odor_concentration

        print(f"Prev Odor Concentration: {prev_odor_concentration}")
        print(f"Current Odor Concentration: {current_odor_concentration}\n")
    
        # ========================== #
        ## Log output
            
        # Log the current step.
        log_entry = {
            "step": step_count,
            "robot_x": robot_x,
            "robot_z": robot_z,
            "robot_yaw": robot_yaw,
            "concentration": current_odor_concentration
        }
        logDF = pd.concat([logDF, pd.DataFrame([log_entry], columns=logDF.columns)], 
                          ignore_index=True)
        
        # # Implement variable step magnitude based on current concentration
        # stepMagnitude = np.interp(current_odor_concentration, [0, 0.3], [0.7, 0.25])


        # ========================== #
        ## Navigation


        source_pos = controller.last_event.metadata["agent"]["position"]
        target_pos = parse_position_string(navKnowledge.iloc[0]["Position"])
        print(f"Target object: {navKnowledge.iloc[0]['objectType']}")
        
        start_node, src_dist = find_nearest_node(graph, source_pos)
        target_node, tgt_dist = find_nearest_node(graph, target_pos)

        # Compute the shortest path from start to end nodes
        path_nodes = nx.dijkstra_path(graph, source=start_node, target=target_node, weight='weight')
        
        path_positions = [graph.nodes[node]['pos'] for node in path_nodes]
        
        try: pos = path_positions[1] # Next position to move to
        except: pos = path_positions[0] # If no path, stay at the current position
        
        # Assume robot_x, robot_y, robot_z and robot_yaw are current values.
        robot_pos = np.array([robot_x, robot_z])
        next_pos = np.array([pos[0], pos[2]])  # use pos[2] for z coordinate

        dir_vector = next_pos - robot_pos

        # Compute target yaw (0° is along +z)
        target_yaw = math.degrees(math.atan2(dir_vector[0], dir_vector[1]))

        relative_yaw = target_yaw - robot_yaw
        if relative_yaw < -180:
            relative_yaw += 360
        elif relative_yaw > 180:
            relative_yaw -= 360

        print(f"Relative yaw: {relative_yaw:.2f}")

        # Rotate towards the target yaw
        if relative_yaw > 0:
            controller.step("RotateRight")
        elif relative_yaw < 0:
            controller.step("RotateLeft")

        print(f"Rotated to target yaw: {target_yaw:.2f}")

        try:
            # Move towards the target position
            controller.step(
                action="Teleport",
                position=dict(x=pos[0], y=pos[1], z=pos[2]),
                rotation=dict(x=0, y=target_yaw, z=0)  # explicitly set rotation
            )
            print(f"Teleported to: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
            
            step_count += 1

            # Check if odor concentration is increasing
            if current_odor_concentration < prev_odor_concentration:
                print("Concentration decreased. Change target.")
                    # Drop the top row from navKnowledge if it exists.
                if navKnowledge:
                    navKnowledge.pop(0)
            else: print("Concentration increased. Continue.")
        except: 
            print("Teleoportation error")
            break

        # Exploration step: update env knowledge
        envKnowledge = visionBranch(yolo_model, envKnowledge, controller)
        navKnowledge = add_goal_similarity(envKnowledge, goal_phrase)
        print(f"New target: {navKnowledge.iloc[0]['objectType']}")
        print(itemDF)
        print("\n")
        
        # Save the current odor concentration as the previous one for the next call.
        fusion_control.prev_odor_concentration = current_odor_concentration

        # Save the current vision frame.
        # frame_filename = f"save/{step_count}.png"
        # cv2.imwrite(frame_filename, controller.last_event.cv2img)
        # print(f"Saved vision frame as {frame_filename}")
        
        # print(f"Robot x: {robot_x}, Robot z: {robot_z}")
        # cv2.imshow("AI2-THOR", controller.last_event.cv2img)
        # cv2.waitKey(int(1000))
        
        time.sleep(0.1)
        
# ==========================
# MAIN FUNCTION
# ==========================
def main():
    stepMagnitude = 0.25
    
    
    # config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    # api_key = config['OPENAI_KEY']
    # gpt_model = config['OPENAI_CHAT_MODEL']
    
    # itemColumns = ["name", "conf", "vizLoc", "glb3DLoc", "goalSimilarity", "searchPriority"]
    itemDF = pd.DataFrame()
    
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
    
    goal = "smoke"
    target_items = ["Microwave"]
    
    # goal = "rotten smell"
    # target_items = ["GarbageCan"]
    
    objects = controller.last_event.metadata["objects"]
    sourcePos = get_objects_centers(objects, target_items)

    # Obtain current scene objects.
    # if target_items == ['Microwave']:
    #     x, y, z = sourcePos[0]
    #     z += 0.5
    #     sourcePos = np.array([[x, y, z]])
    # elif target_items == ['GarbageCan']:
    #     x, y, z = sourcePos[0]
    #     x += 0.25
    #     sourcePos = np.array([[x, y, z]])
        

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
        goal_phrase=goal,
        dist_threshold=1.2,
        stepMagnitude=stepMagnitude
    )

if __name__ == "__main__":
    main()
