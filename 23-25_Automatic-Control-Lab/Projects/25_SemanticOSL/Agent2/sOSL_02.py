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
    # Using the convention: optical axis is z, x is to the right, and y is upward.
    x_cam = d * np.sin(theta_h) * np.cos(theta_v)
    y_cam = d * np.sin(theta_v)  # if image v increases downward, you might need to flip sign
    z_cam = d * np.cos(theta_h) * np.cos(theta_v)
    
    # Incorporate robot's rotation:
    robot_yaw_deg = controller.last_event.metadata["agent"]["rotation"]["y"]
    # Rotate x_cam and z_cam from the camera frame to align with global axes.
    x_rot, z_rot = rotate_point(x_cam, z_cam, robot_yaw_deg)

    # Compute global coordinates by applying the robot's global position.
    x_global = controller.last_event.metadata["agent"]["position"]["x"] - x_rot
    y_global = controller.last_event.metadata["agent"]["position"]["y"] - y_cam
    z_global = controller.last_event.metadata["agent"]["position"]["z"] - z_rot
    
    return x_global, y_global, z_global


def visionBranch(model, itemDF, controller, confThr=0.1):
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
    Computes the cosine similarity between each object's type and a goal phrase,
    adds the similarity as a 'goalSim' key, and sorts the table by (Conf * goalSim)
    in descending order.
    
    Parameters:
        table (list): A list of dictionaries representing objects with keys 'objectType', 'Conf', and 'Position'.
        goal_phrase (str): The phrase to compare against (default: "burning smell").
    
    Returns:
        list: The updated and sorted table with an added 'goalSim' column.
    """
    # Load the pre-trained Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute the embedding for the goal phrase once
    goal_embedding = model.encode(goal_phrase, convert_to_tensor=True)
    
    # For each object in the table, compute cosine similarity between the object type and the goal phrase.
    for row in table:
        # Encode the object type to obtain its embedding
        object_embedding = model.encode(row["objectType"], convert_to_tensor=True)
        # Compute cosine similarity
        cosine_sim = util.pytorch_cos_sim(object_embedding, goal_embedding).item()
        # Add the computed similarity to the row
        row["goalSim"] = cosine_sim
    
    # Sort the table by the product of 'Conf' and 'goalSim' in descending order
    table.sort(key=lambda x: x["Conf"] * x["goalSim"], reverse=True)
    return table

def fusion_control(controller, itemDF, yolo_model, source_position, 
                   save_path="itemDF.csv", step_threshold=50, max_time=150, goal_phrase="", 
                   dist_threshold=1.0, stepMagnitude=0.5):
    """
    Automatic control loop.
    Each iteration:
      - Vision branch provides environment knowledge (object detections with 3D coordinates and goal similarity).
      - Olfaction branch provides odor concentration.
      - Fusion branch combines vision and olfaction data.
          * Approach the object with highest goal similarity.
          * If odor concentration decreases while distance to the object decreases,
            discard that object and approach the next best target.
          * Otherwise, if the robot reaches within a threshold distance to the object, terminate.
      - Logs the time, robot position (x, z), robot yaw, etc.
    """

    step_count = 1    
    start_time = time.time()
    logDF = pd.DataFrame(columns=["step", "robot_x", "robot_z", "robot_yaw", 
                                  "target_object", "concentration"])
    
    print("Fusion control active. Executing actions until timeout or target reached.")
    
    # Get reachable positions and generate a graph of reachable positions.
    positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    graph = create_graph_from_positions(positions, threshold=0.3)

    # Initialize previous odor concentration for later comparisons.
    fusion_control.prev_odor_concentration = None

    while True:
        print("\n=============================")
        print("New Step")
        print("=============================\n")
        print(f"Steps: {step_count}/{step_threshold}")

        # ========================== #
        ## Vision Branch: update environment knowledge
        # Update itemDF with new detections.
        itemDF = visionBranch(yolo_model, itemDF, controller)
        print("Updated itemDF (environment knowledge):")
        print(itemDF)

        # Convert the DataFrame to a list of dictionaries for goal similarity.
        envKnowledge = itemDF.to_dict('records')
        if envKnowledge:  
            navKnowledge = add_goal_similarity(envKnowledge, goal_phrase)
            print("Navigation Knowledge (sorted by goal similarity):")
            print(navKnowledge)
        else:
            navKnowledge = []
            print("No objects detected in vision branch.")

        # ========================== #
        ## Ground truth: distance to source position.
        if source_position.size > 0:
            distances = [get_distance_to_source(controller, center) for center in source_position]
            min_distance = min(distances)
        else:
            min_distance = float('inf')
        print(f"Current minimum distance to target: {min_distance:.2f}")

        # Termination condition.
        if min_distance < dist_threshold:
            print(f"Robot is within {dist_threshold} of the target. Mission accomplished!")
            logDF.to_csv("save/trajectory_log.csv", index=False)
            break
        
        if step_count >= step_threshold:
            print(f"Step limit of {step_threshold} reached. Saving log and exiting.")
            logDF.to_csv("save/trajectory_log.csv", index=False)
            break

        # Retrieve robot's current pose.
        agent_meta = controller.last_event.metadata["agent"]
        robot_x = agent_meta["position"].get("x", None)
        robot_z = agent_meta["position"].get("z", None)  # ground plane coordinate
        robot_yaw = agent_meta["rotation"].get("y", None)
        
        # ========================== #
        ## Olfaction Branch: odor concentration
        current_odor_concentration = olfactionBranch(source_position, controller)
        prev_odor_concentration = fusion_control.prev_odor_concentration
        if prev_odor_concentration is None:
            prev_odor_concentration = current_odor_concentration
        print(f"Prev Odor Concentration: {prev_odor_concentration}")
        print(f"Current Odor Concentration: {current_odor_concentration}\n")
    
        # ========================== #
        ## Log output
        log_entry = {
            "step": step_count,
            "robot_x": robot_x,
            "robot_z": robot_z,
            "robot_yaw": robot_yaw,
            "concentration": current_odor_concentration,
            "target_object": navKnowledge[0]["objectType"] if navKnowledge else "None"
        }
        logDF = pd.concat([logDF, pd.DataFrame([log_entry], columns=logDF.columns)], 
                          ignore_index=True)
        
        # ========================== #
        ## Navigation: select target and move.
        source_pos = controller.last_event.metadata["agent"]["position"]
        # Use the highest goal similarity object (first in sorted navKnowledge).
        if navKnowledge:
            target_pos = parse_position_string(navKnowledge[0]["Position"])
            print(f"Target object: {navKnowledge[0]['objectType']}")
        else:
            print("No navigation target available. Exiting loop.")
            break
        
        start_node, src_dist = find_nearest_node(graph, source_pos)
        target_node, tgt_dist = find_nearest_node(graph, target_pos)

        path_nodes = nx.dijkstra_path(graph, source=start_node, target=target_node, weight='weight')
        path_positions = [graph.nodes[node]['pos'] for node in path_nodes]
        try: 
            pos = path_positions[1]  # next position to move to
        except: 
            pos = path_positions[0]
        
        robot_pos = np.array([robot_x, robot_z])
        next_pos = np.array([pos[0], pos[2]])  # using pos[2] for z coordinate

        dir_vector = next_pos - robot_pos
        target_yaw = math.degrees(math.atan2(dir_vector[0], dir_vector[1]))
        relative_yaw = target_yaw - robot_yaw
        if relative_yaw < -180:
            relative_yaw += 360
        elif relative_yaw > 180:
            relative_yaw -= 360

        print(f"Relative yaw: {relative_yaw:.2f}")

        # Rotate towards target.
        if relative_yaw > 0:
            controller.step("RotateRight")
        elif relative_yaw < 0:
            controller.step("RotateLeft")
        print(f"Rotated to target yaw: {target_yaw:.2f}")

        try:
            controller.step(
                action="Teleport",
                position=dict(x=pos[0], y=pos[1], z=pos[2]),
                rotation=dict(x=0, y=target_yaw, z=0)
            )
            print(f"Teleported to: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
            step_count += 1

            # If odor concentration decreases, drop the current top navigation target.
            if current_odor_concentration < prev_odor_concentration:
                print("Odor concentration decreased. Changing target.")
                if navKnowledge:
                    navKnowledge.pop(0)
            else:
                print("Odor concentration increased. Continuing with current target.")
        except Exception as e:
            print("Teleportation error:", e)
            break

        # Update previous odor concentration for next iteration.
        fusion_control.prev_odor_concentration = current_odor_concentration

        time.sleep(0.1)

        

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
            
    itemColumns = ["objectType", "Conf", "Position"]
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
    
    goal = "rotten smell"
    target_items = ["GarbageCan"]
    
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

    # Garbage Start Pos 2: upper left corner
    controller.step(
        action="Teleport",
        position=dict(x=2, y=0.9, z=-1.5),
        rotation=dict(x=0, y=180, z=0),
    )

    controller.step(
        "MoveAhead",
        moveMagnitude=0.01
    )


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
