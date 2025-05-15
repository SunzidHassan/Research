import time
import math
import numpy as np
import pandas as pd
import os
import io
from PIL import Image
import cv2
import re
import networkx as nx

from ai2thor.controller import Controller
from ultralytics import YOLO
# from openai import OpenAI  # if needed
from sentence_transformers import SentenceTransformer, util

from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

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

# ==========================
# Infotaxis FUNCTIONS
# ==========================

def parse_position_string(pos_str):
    # Expecting a string like "x, y, z"
    return np.array([float(val.strip()) for val in pos_str.split(',')])

def likelihood(current_odor_concentration, expected, sigma_noise):
    return (
        np.exp(-((current_odor_concentration - expected) ** 2) / (2 * sigma_noise ** 2))
        / (np.sqrt(2 * np.pi) * sigma_noise)
    )

def entropy(prob_map):
    eps = 1e-12
    return -np.sum(prob_map * np.log(prob_map + eps))

def world_to_grid(x, z, x_points, z_points):
    col = (np.abs(x_points - x)).argmin()
    row = (np.abs(z_points - z)).argmin()
    return np.array([row, col])

def grid_to_world(pos, x_points, z_points):
    row, col = pos
    return np.array([x_points[col], z_points[row]])


# ==========================
# OLFACTION FUNCTIONS
# ==========================

def gaussian_plume(x, z, source, sigma=2.0):
    Sx, Sz = source
    dx = x - Sx
    dz = z - Sz
    return np.exp(-(dx**2 + dz**2) / (2 * sigma**2))


def olfactionBranch(source, controller, sigma_plume=2.0, sigma_noise=0.1):
    robot_x, robot_y, robot_z = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
    return gaussian_plume(robot_x, robot_z, source, sigma_plume) + np.random.normal(0, sigma_noise)

class Infotaxis:
    def __init__(self, pos, src_pos, x_points, z_points, sigma_plume, sigma_noise, reachable_positions):
        self.pos = pos
        self.src_pos = src_pos
        self.x_points = x_points
        self.z_points = z_points
        self.window_x = len(x_points)
        self.window_z = len(z_points)
        # Start with a uniform belief
        self.prob_map = np.full((self.window_z, self.window_x), 1.0 / (self.window_z * self.window_x))
        self.sigma_plume = sigma_plume
        self.sigma_noise = sigma_noise
        self.reachable_positions = reachable_positions  # list of (x, z)


    def update_belief(self, current_odor_concentration, robot_x, robot_z, smooth_sigma=1.0):
        # 1) Bayesian update (pointwise multiplication)
        for iz, z in enumerate(self.z_points):
            for ix, x in enumerate(self.x_points):
                expected = gaussian_plume(robot_x, robot_z, (x, z), self.sigma_plume)
                self.prob_map[iz, ix] *= likelihood(current_odor_concentration, expected, self.sigma_noise)

        # 2) Normalize once
        self.prob_map /= self.prob_map.sum()

        # 3) Apply Gaussian smoothing
        if smooth_sigma > 0.0:
            self.prob_map = gaussian_filter(self.prob_map, sigma=smooth_sigma, mode='reflect')

            # 4) Re-normalize to ensure it remains a valid probability distribution
            self.prob_map /= np.sum(self.prob_map)

    def best_move(self, robot_x, robot_z, step_size):
        best_expected_ent = np.inf
        best_move = None

        for dz in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                new_x = robot_x + dx * step_size
                new_z = robot_z + dz * step_size

                if not self.is_valid_move(new_x, new_z):
                    continue

                # Hypothetical measurement & posterior entropy
                expected_meas = self.compute_expected_measurement(new_x, new_z)
                new_ent = self.compute_expected_entropy(new_x, new_z, expected_meas)

                if new_ent < best_expected_ent:
                    best_expected_ent = new_ent
                    best_move = (new_x, new_z)

        return best_move

    def is_valid_move(self, x, z, tol=1e-4):
        return any(abs(rx - x) < tol and abs(rz - z) < tol for rx, rz in self.reachable_positions)

    def compute_expected_measurement(self, x, z):
        xgrid, zgrid = np.meshgrid(self.x_points, self.z_points)
        plume_vals = gaussian_plume(x, z, (xgrid, zgrid), self.sigma_plume)
        return np.sum(self.prob_map * plume_vals)

    def compute_expected_entropy(self, x, z, expected_meas):
        xgrid, zgrid = np.meshgrid(self.x_points, self.z_points)
        expected_vals = gaussian_plume(x, z, (xgrid, zgrid), self.sigma_plume)
        new_belief = self.prob_map * likelihood(expected_meas, expected_vals, self.sigma_noise)
        new_belief /= new_belief.sum()
        return entropy(new_belief)

    def is_source_found(self, robot_x, robot_z, threshold=1.25):
        """
        Checks if the robot is within `threshold` distance of the odor source in world coordinates.
        Also prints the distance at each check.
        """
        src_xy = grid_to_world(self.src_pos, self.x_points, self.z_points)
        distance = np.linalg.norm(np.array([robot_x, robot_z]) - src_xy)
        # print(f"Distance to source: {distance:.3f}")
        return distance <= threshold

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


def initialize_envKnowledge(controller, model, itemDF, probMap, x_points, z_points, confThr=0.3, nav_mode = 'f'):
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
    # itemDF = add_goal_similarity(itemDF, "burning smell", probMap, x_points, z_points, nav_mode)
    return itemDF


def add_goal_similarity(itemDF, goal_phrase, probMap, x_points, z_points, nav_mode='f'):
    """
    Updates itemDF with similarity columns:
      - visionSim: detection confidence
      - olfactionSim: Infotaxis belief from probMap
      - langSim: textual similarity to goal phrase
    Then calculates a combined similarity: goalSim = langSim * visionSim * olfactionSim.

    Parameters:
        itemDF (pd.DataFrame): Must contain columns:
            - objectType (string)
            - Conf (float)
            - Position (string) in 'x, y, z' format
        goal_phrase (str): User's textual goal
        probMap (2D np.ndarray): Infotaxis belief map
        x_points, z_points (np.ndarray): World-to-grid coordinate arrays
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    goal_embedding = model.encode(goal_phrase, convert_to_tensor=True)

    # Initialize columns
    itemDF["visionSim"] = itemDF["Conf"]
    itemDF["olfactionSim"] = np.nan
    itemDF["langSim"] = np.nan
    itemDF["goalSim"] = np.nan

    for idx, row in itemDF.iterrows():
        object_type = row["objectType"]

        # Language similarity
        object_embedding = model.encode(object_type, convert_to_tensor=True)
        lang_similarity = util.pytorch_cos_sim(object_embedding, goal_embedding).item()
        itemDF.at[idx, "langSim"] = lang_similarity

        # Convert position from world to grid coordinates
        pos_str = row["Position"]
        x_world, _, z_world = map(float, pos_str.split(','))
        grid_row, grid_col = world_to_grid(x_world, z_world, x_points, z_points)

        # Olfaction similarity
        olf_val = probMap[grid_row, grid_col]
        itemDF.at[idx, "olfactionSim"] = olf_val

        # Combined goal similarity
        vision_sim = row["visionSim"]
        if nav_mode == "f":
            combined_sim = lang_similarity * olf_val
        elif nav_mode == "v":
            combined_sim = lang_similarity
        elif nav_mode == "o":
            combined_sim = olf_val
        itemDF.at[idx, "goalSim"] = combined_sim

    # Sort by combined similarity
    itemDF.sort_values(by="goalSim", ascending=False, inplace=True)

    # Print the x, z coordinate of highest belief map value
    max_index = np.unravel_index(np.argmax(probMap), probMap.shape)
    max_x = x_points[max_index[1]]
    max_z = z_points[max_index[0]]
    print(f"Highest belief map coordinate: x={max_x}, z={max_z}")

    return itemDF


def generate_heatmap(df, x_points, z_points, weight_key='goalSim', sigma=1.0, upsample=8):
    """
    Build a normalized [0–1] heatmap on the grid defined by x_points, z_points,
    using df rows weighted by df[weight_key], then upsample by 'upsample' factor.
    """
    H, W = len(z_points), len(x_points)
    heatmap = np.zeros((H, W), dtype=float)

    for _, row in df.iterrows():
        x_w, _, z_w = map(float, row["Position"].split(','))
        r, c = world_to_grid(x_w, z_w, x_points, z_points)
        heatmap[r, c] += float(row.get(weight_key, 0.0))

    # smooth then max-normalize so peak → 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma, mode='reflect')
    heatmap /= (heatmap.max() + 1e-8)

    # upsample for higher resolution
    if upsample > 1:
        heatmap = cv2.resize(
            heatmap, 
            (W * upsample, H * upsample), 
            interpolation=cv2.INTER_LINEAR
        )
    return heatmap

# ==========================
# Nav Functions
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

# ==========================
# CONTROL LOOP
# ==========================

def fusion_control(controller, itemDF, yolo_model, source_position, 
                 save_path="itemDF.csv", step_threshold = 50, max_time=150, goal_phrase="", 
                 dist_threshold=1.0, stepMagnitude=0.5,
                 infotaxis_agent=None, x_points=None, z_points=None, nav_mode = 'f', step_size=0.25):
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
    entropies = []
    logDF = pd.DataFrame(columns=["step", "robot_x", "robot_z", "robot_yaw", 
                                  "target_object", "concentration"])
    # Retrieve robot's current pose.
    agent_meta = controller.last_event.metadata["agent"]
    robot_x = agent_meta["position"].get("x", None)
    robot_z = agent_meta["position"].get("z", None)  # using z for ground plane coordinate
    robot_yaw = agent_meta["rotation"].get("y", None)

    print("Fusion control active. Executing actions until timeout or target reached.")
    
    # ========================== #
    ## Vision Branch: environment knowledge -> coordiante
    # Get environment knowledge
    # 1) Update Infotaxis belief
    Sx, Sy, Sz = source_position[0]
    current_odor_concentration = olfactionBranch((Sx, Sz), controller)
    infotaxis_agent.update_belief(current_odor_concentration, robot_x, robot_z)

    # 2) Extract the updated belief map
    probMap = infotaxis_agent.prob_map  # The agent's final distribution

    # 3) Use the belief map where needed
    envKnowledge = initialize_envKnowledge(controller, yolo_model, itemDF, probMap, x_points=x_points, z_points=z_points, confThr=0.5, nav_mode=nav_mode)
    navKnowledge = add_goal_similarity(envKnowledge, goal_phrase, probMap, x_points, z_points, nav_mode = nav_mode)

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

            # Log the final step before breaking
            log_entry = {
                "step": step_count,
                "robot_x": robot_x,
                "robot_z": robot_z,
                "robot_yaw": robot_yaw,
                "concentration": current_odor_concentration
            }
            logDF = pd.concat([logDF, pd.DataFrame([log_entry], columns=logDF.columns)], ignore_index=True)

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
        current_odor_concentration = olfactionBranch((Sx, Sz), controller)
        
        prev_odor_concentration = getattr(fusion_control, "prev_odor_concentration", None)
        
        # If no previous value exists, initialize it.
        if prev_odor_concentration is None:
           prev_odor_concentration = current_odor_concentration

        print(f"Prev Odor Concentration: {prev_odor_concentration}")
        print(f"Current Odor Concentration: {current_odor_concentration}\n")


        # Step 2: update belief
        infotaxis_agent.update_belief(current_odor_concentration, robot_x, robot_z)
        current_ent = entropy(infotaxis_agent.prob_map)
        print(f"[Step {step_count}] Infotaxis Entropy: {current_ent:.4f}")
        entropies.append(current_ent)

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
        
        # Compute midpoint between robot and target
        robot_pos = np.array([robot_x, agent_meta["position"]["y"], robot_z])
        target_pos_full = np.array([target_pos[0], agent_meta["position"]["y"], target_pos[2]])
        midpoint = (robot_pos + target_pos_full) / 2.0

        # Find the nearest node in the graph to the midpoint
        midpoint_node, _ = find_nearest_node(graph, midpoint.tolist())
        pos = graph.nodes[midpoint_node]["pos"]
        
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

        except: 
            print("Teleoportation error")
            break

        # Exploration step: update env knowledge
        probMap = infotaxis_agent.prob_map  # The agent's final distribution
        envKnowledge = visionBranch(yolo_model, envKnowledge, controller)
        navKnowledge = add_goal_similarity(envKnowledge, goal_phrase, probMap, x_points, z_points, nav_mode = nav_mode)
        
        
        print(f"New target: {navKnowledge.iloc[0]['objectType']}")
        # Print only the relevant columns. For example:
        if nav_mode == 'f':
            columns_of_interest = ["objectType", "Position", "olfactionSim", "langSim", "goalSim"]
        elif nav_mode == 'v':
            columns_of_interest = ["objectType", "Position", "langSim"]
        elif nav_mode == 'o':
            columns_of_interest = ["objectType", "Position", "olfactionSim"]
            
        print(navKnowledge[columns_of_interest])
        # Print the x, z coordinate of highest belief map value
        max_index = np.unravel_index(np.argmax(probMap), probMap.shape)
        max_x = x_points[max_index[1]]
        max_z = z_points[max_index[0]]
        print(f"Highest belief map coordinate: x={max_x}, z={max_z}")

        print("\n")
        
        # Save the current odor concentration as the previous one for the next call.
        fusion_control.prev_odor_concentration = current_odor_concentration


        # ========================== #
        # Save infotaxis belief map
        # Save side-by-side infotaxis and vision maps

                
        # 1) Infotaxis map
        H, W = len(z_points), len(x_points)
        infomap_raw = probMap / (probMap.max() + 1e-8)
        # upsample to (width, height)
        infomap_up = cv2.resize(infomap_raw, (W*2, H*2), interpolation=cv2.INTER_LINEAR)
        # flip vertically so that array[0,:] (z_min) is at the bottom
        infomap_up = cv2.flip(infomap_up, 0)
        infomap_img   = (infomap_up * 255).astype(np.uint8)
        infomap_color = cv2.applyColorMap(infomap_img, cv2.COLORMAP_JET)

        # 2) Vision map (langSim)
        H, W = len(z_points), len(x_points)   # same dims
        vision_raw = generate_heatmap(navKnowledge, x_points, z_points,
                                      weight_key='langSim', sigma=1)
        vision_up  = cv2.resize(vision_raw,  (W*2, H*2), interpolation=cv2.INTER_LINEAR)
        vision_up  = cv2.flip(vision_up, 0)
        vision_img = (vision_up * 255).astype(np.uint8)
        vision_color = cv2.applyColorMap(vision_img, cv2.COLORMAP_HOT)

        # 3) goalSim map
        goal_raw = generate_heatmap(navKnowledge, x_points, z_points,
                                    weight_key='goalSim', sigma=1)
        goal_up  = cv2.resize(goal_raw,   (W*2, H*2), interpolation=cv2.INTER_LINEAR)
        goal_up  = cv2.flip(goal_up, 0)
        goal_img = (goal_up * 255).astype(np.uint8)
        goal_color  = cv2.applyColorMap(goal_img, cv2.COLORMAP_PLASMA)

                        
            
        # Convert to RGB
        inf_rgb  = cv2.cvtColor(infomap_color, cv2.COLOR_BGR2RGB)
        vis_rgb  = cv2.cvtColor(vision_color,  cv2.COLOR_BGR2RGB)
        goal_rgb = cv2.cvtColor(goal_color,    cv2.COLOR_BGR2RGB)

        # Figure setup
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        maps   = [inf_rgb, vis_rgb, goal_rgb]
        titles = ['Infotaxis', 'Vision (langSim)', 'GoalSim']

        # Compute bounds
        xmin, xmax = min(x_points), max(x_points)
        zmin, zmax = min(z_points), max(z_points)
        # Reverse the z‐extent so that zmax is at the top
        extent = [xmin, xmax, zmax, zmin]

        # Integer tick marks
        xticks = list(range(math.floor(xmin), math.ceil(xmax) + 1))
        zticks = list(range(math.floor(zmin), math.ceil(zmax) + 1))

        for ax, img, title in zip(axes, maps, titles):
            # Put row 0 at the top, and use the reversed z‐extent
            ax.imshow(img, origin='upper', extent=extent)
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_xticks(xticks)
            ax.set_yticks(zticks)
            ax.grid(True, linestyle='--', linewidth=0.5)
            # Data limits match the extent (no further inversion)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(zmax, zmin)

        plt.tight_layout()
        out_fname = f"save/maps_all_{step_count}_x_{robot_x:.2f}_z_{robot_z:.2f}.png"
        fig.savefig(out_fname, dpi=150)
        plt.close(fig)

        print(f"Saved combined map (aligned to trajectory) as {out_fname}")

        # Save egocentric RGB frame
        frame_bgr = controller.last_event.cv2img  # AI2-THOR gives frame in BGR format
        cv2.imwrite(f"save/frame_{step_count:03d}_x_{robot_x:.2f}_z_{robot_z:.2f}.png", frame_bgr)

        
        time.sleep(0.1)

# ==========================
# MAIN FUNCTION
# ==========================
def main():
    # mode = input("Select navigation mode (f: fusion, v: vision only, o: olfaction only): ").strip().lower()
    mode = 'f'
    while mode not in ['f', 'v', 'o']:
        mode = input("Invalid input. Please enter 'f', 'v', or 'o': ").strip().lower()


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
    
    scene_bounds = controller.last_event.metadata["sceneBounds"]

    # Convert corner points to a numpy array
    corner_points = np.array(scene_bounds["cornerPoints"])
    # Round them to nearest 0.25 increments (if desired)
    corner_points = np.floor(corner_points / 0.25) * 0.25

    # Extract x and z coords
    x_coords = corner_points[:, 0]
    z_coords = corner_points[:, 2]
    # Determine min and max
    x_min, x_max = np.round(x_coords.min(), 2), np.round(x_coords.max(), 2)
    z_min, z_max = np.round(z_coords.min(), 2), np.round(z_coords.max(), 2)

    # Step size for the environment
    step_size = 0.25

    # Create the coordinate arrays
    x_points = np.arange(x_min, x_max + step_size, step_size)
    z_points = np.arange(z_min, z_max + step_size, step_size)

    print(f'X bounds: {x_min}, {x_max}')
    print(f'Z bounds: {z_min}, {z_max}')

    # (1) Retrieve the odor source from iTHOR objects
    goal = "smoke"
    target_items = ["Microwave"]
    
    # goal = "rotten smell"
    # target_items = ["Fridge"]
    
    objects = controller.last_event.metadata["objects"]
    sourcePos = get_objects_centers(objects, target_items)
    x, y, z = sourcePos[0]

    # (2) Environment parameters
    sigma_plume = 2.0
    sigma_noise = 0.1

    # (3) Retrieve the robot's start info
    agent_pos = controller.last_event.metadata["agent"]["position"]
    robot_x = agent_pos["x"]
    robot_z = agent_pos["z"]

    reachable = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    reachable_positions = [(pos["x"], pos["z"]) for pos in reachable]

    # (4) Setup the Infotaxis agent
    src_grid_pos = world_to_grid(x, z, x_points, z_points)
    start_grid_pos = world_to_grid(robot_x, robot_z, x_points, z_points)
    infotaxis_agent = Infotaxis(
    pos = start_grid_pos,
    src_pos= src_grid_pos,
    x_points= x_points,
    z_points= z_points,
    sigma_plume= sigma_plume,
    sigma_noise= sigma_noise,
    reachable_positions=reachable_positions
    )

    # Obtain current scene objects.
    if target_items == ['Microwave']:
        x, y, z = sourcePos[0]
        z += 0.5
        sourcePos = np.array([[x, y, z]])
    elif target_items == ['Fridge']:
        x, y, z = sourcePos[0]
        x += 0.25
        sourcePos = np.array([[x, y, z]])
        

    # # Microwave Starting position 1
    controller.step(
        action="Teleport",
        position=dict(x=1.5, y=0.9, z=1.5),
        rotation=dict(x=0, y=180, z=0)
    )

    controller.step(
        "MoveAhead",
        moveMagnitude=0.01
    )
    
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
    
    
    # Garbage Start Pos 1: facing back to the garbage bin
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
        # dist_threshold=1.2,
        dist_threshold=0.55,
        stepMagnitude=stepMagnitude,
        infotaxis_agent=infotaxis_agent,
        x_points=x_points,
        z_points=z_points,
        nav_mode = mode
    )

if __name__ == "__main__":
    main()
