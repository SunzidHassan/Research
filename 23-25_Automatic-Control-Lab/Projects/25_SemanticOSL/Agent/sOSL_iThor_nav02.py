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

from ai2thor.controller import Controller
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util

# nav01
from sOSL_iThor_gndTruth import get_objects_centers  # Imported helper

# nav02
from openai import OpenAI  # if needed
from gptPayload import payload
# llmPrompt is now integrated into gptNav

# ==========================
# SENSOR FUNCTIONS
# ==========================

def get_field_value(robot_pos, sourcesPos, q_s=2000, D=10, U=0, tau=1000, del_t=1, psi_deg=0):
    """
    Computes the odor field value at a given robot position as the sum of contributions
    from one or more odor sourcesPos.
    
    Parameters:
        robot_pos (ndarray): A NumPy array of shape (2,) representing the robot's [x, y] position.
        sourcesPos (ndarray): A NumPy array of shape (n,2) where each row is [x_s, y_s] for a source.
        q_s (float): Source strength.
        D (float): Diffusion coefficient.
        U (float): Advection velocity (set to 0 if no airflow).
        tau (float): Time or scaling parameter.
        del_t (float): Time step.
        psi_deg (float): Angle in degrees for rotation (direction of advection; irrelevant if U==0).
    
    Returns:
        (float): The computed field value.
    """
    psi = math.radians(psi_deg)
    lambd = math.sqrt((D * tau) / (1 + (tau * U**2) / (4 * D)))
    total = 0.0
    x, y = robot_pos  # Unpack the robot position
    for source in sourcesPos:
        x_s, y_s = source  # Unpack the source coordinates
        delta_x, delta_y = x - x_s, y - y_s
        r = np.hypot(delta_x, delta_y)
        if r == 0:
            contribution = - (r / lambd) * del_t
        else:
            rotated_y = -delta_x * math.sin(psi) + delta_y * math.cos(psi)
            term1 = (q_s / (4 * math.pi * D * r)) * math.exp(-rotated_y * U / (2 * D))
            term2 = - (r / lambd) * del_t
            contribution = term1 + term2
        total += contribution
    return total

def olfactionBranch(sourcePos, controller, q_s=2000, D=10, U=0, tau=1000, del_t=1, psi_deg=0):
    """
    Computes odor concentration based on the odor source position and the robot's current position.
    Note: get_field_value expects a 2D position (x, y), so we pass the robot's x and z, and source's x and z.
    If sourcePos is invalid, returns 0.
    """
    robot_pos = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
    # Ensure sourcePos is a numpy array with at least 3 elements.
    if sourcePos is None or (hasattr(sourcePos, "shape") and sourcePos.shape[0] < 3):
        return 0
    plumeConcentration = int(get_field_value(robot_pos[[0,2]], sourcePos[[0,2]], q_s=q_s, D=D, U=U, tau=tau, del_t=del_t, psi_deg=psi_deg))
    return plumeConcentration

def objDetector(itemDF, controller, objDetectorModel):
    """
    Updates itemDF with YOLO object detection results and depth estimation.
    """
    itemList = []
    results = objDetectorModel(controller.last_event.frame)
    depthFrame = np.array(controller.last_event.depth_frame)
    
    for box in results[0].boxes:
        x, y, w, h = box.xywh[0]
        confidence = box.conf[0].item()
        class_name = objDetectorModel.names[int(box.cls[0].item())]
        x_rounded, y_rounded = round(x.item()), round(y.item())
        if 0 <= y_rounded < depthFrame.shape[0] and 0 <= x_rounded < depthFrame.shape[1]:
            depth_value = float(depthFrame[y_rounded, x_rounded])
        else:
            depth_value = np.nan
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

def actionTable(itemDF, conf_thres=0.5):
    """
    Creates an action DataFrame based on the current itemDF detections.
    
    The action table has two columns:
      - Action: One of "Forward", "Turn Left", "Turn Right"
      - Obstacle: Indicates if an obstacle is present ("Yes" or "No")
    
    Object names are not included.
    """
    actions = ["Forward", "Turn Left", "Turn Right"]
    table = {"Action": actions, "Obstacle": ["" for _ in actions]}
    return pd.DataFrame(table)

def obstacleTable(actionDF, controller, threshold_distance):
    """
    Updates the Obstacle column in actionDF by analyzing the depth frame.
    Uses a central region to determine if there is an obstacle in front.
    """
    depth_frame = np.array(controller.last_event.depth_frame)
    forward_region = depth_frame[:, 120:180]
    forward_min = np.min(forward_region)
    
    left_obstacle = "No"
    forward_obstacle = "Yes" if forward_min < threshold_distance else "No"
    right_obstacle = "No"
    
    for idx, row in actionDF.iterrows():
        if row["Action"] == "Forward":
            actionDF.at[idx, "Obstacle"] = forward_obstacle
        elif row["Action"] == "Turn Left":
            actionDF.at[idx, "Obstacle"] = left_obstacle
        elif row["Action"] == "Turn Right":
            actionDF.at[idx, "Obstacle"] = right_obstacle
    
    return actionDF

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
# GPT NAVIGATION
# ==========================
def payload(api_key, prompt, image, model):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload_data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 300,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload_data)
    response_json = response.json()
    if "choices" not in response_json or not response_json["choices"]:
        print("Error in GPT API response:")
        print(response_json)
        raise ValueError("GPT API response does not contain 'choices'.")
    return response_json["choices"][0]["message"]["content"]

def llmPrompt(delimiter, goal, actionDF_str, plume):
    """
    Generates a prompt for the GPT model to select one action from the action table.
    """
    task = f"""
    Your task is to select the best action for a mobile robot to move towards the source of {goal}.
    You are provided with an image and an Action Table that summarizes the robot's current surroundings.
    
    The image includes:
    - Robot's current egocentric view.

    The table includes:
      - **Action**: The potential action (e.g., Move Forward, Rotate Left, Rotate Right).
      - **Obstacle**: Indicates if an obstacle is present ("Yes" or "No") in that direction.
    
    Additionally, the current chemical concentration measured (plume value) is {plume}.
    
    The rules are:
      1. If there are no obstacles in the Forward direction and the image suggests the presence of the target odor source, the robot should move forward.
      2. If there is no object related to the {goal} present, the robot should turn left to explore a new view.
      3. If the Forward direction is blocked, then consider turning left or right if that direction is clear.
      4. Only one action should be selected.
    """
    
    actionInstructions = """
    Action 1: Move forward. (Action_id = 1)
    Action 2: Rotate right. (Action_id = 2)
    Action 3: Rotate left. (Action_id = 3)
    """
    
    reasoningOutputInstructions = f"""
    Your response should follow this format:
    <reasoning>
    (Provide a brief step-by-step explanation of your decision)
    Selected Action:{delimiter} <Output only one Action_id as an integer>
    Use {delimiter} to separate each step.
    """
    
    prompt = f"""
    {delimiter} Task:
    {task}
    {delimiter} Available Actions:
    {actionInstructions}
    {delimiter} Current Action Table:
    {actionDF_str}
    {delimiter} Output Instructions:
    {reasoningOutputInstructions}
    """
    
    return prompt

def gptNav(controller, api_key, gpt_model, goal, actionDF, sourcePos):
    delimiter = "#####"
    actionDF_str = actionDF.to_string(index=False)
    plume = str(olfactionBranch(sourcePos, controller))
    prompt = llmPrompt(delimiter, goal, actionDF_str, plume)
    image_array = controller.last_event.frame
    compressed_bytes = compress_image(image_array, size=(64, 64))
    image_base64 = base64.b64encode(compressed_bytes).decode('utf-8')
    response = payload(api_key, prompt, image_base64, gpt_model)
    return response

# ==========================
# AUTOMATIC CONTROL LOOP
# ==========================
def auto_control(controller, itemDF, yolo_model, api_key, gpt_model, target_items, save_path="itemDF.csv", max_time=150, goal="food burning smell", dist_threshold=0.5):
    """
    Automatic control loop.
    Each iteration:
      1. Updates vision using YOLO.
      2. Builds the action table (with obstacles) containing only Action and Obstacle.
      3. Extracts scene objects and computes target object centers.
      4. Computes the minimum distance from the robot to any target object.
         If the distance is below dist_threshold, the loop terminates.
      5. Calls GPT to select the best action based solely on the attached (compressed) image.
         GPT will decide the action by analyzing the image.
      6. Logs the time, robot position (x, y), robot yaw, and GPT decision.
    """
    start_time = time.time()
    logDF = pd.DataFrame(columns=["time", "robot x", "robot y", "robot yaw", "gpt decision"])
    
    print("Automatic control active. Executing GPT-selected actions until timeout or target reached.")
    print(f"Timeout: {max_time} seconds. Termination if distance < {dist_threshold}.")
    
    while True:
        print("=============================")
        print("New Step")
        print("=============================")
        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        
        if elapsed_time > max_time:
            print(f"\nTime limit of {max_time} seconds reached. Saving itemDF and log, then exiting.")
            _save_itemDF(itemDF, save_path)
            logDF.to_csv("save/trajectory_log.csv", index=False)
            break
        
        # Obtain current scene objects.
        objects = controller.last_event.metadata["objects"]
        centers = get_objects_centers(objects, target_items)
        
        if centers.size > 0:
            distances = [get_distance_to_source(controller, center) for center in centers]
            min_distance = min(distances)
            if centers.ndim == 1:
                sourcePos = centers
            else:
                sourcePos = centers[np.argmin(distances)]
        else:
            min_distance = float('inf')
            sourcePos = None
        
        print(f"Current minimum distance to target: {min_distance:.2f}")
        if min_distance < dist_threshold:
            print(f"Robot is within {dist_threshold} of the target. Mission accomplished!")
            _save_itemDF(itemDF, save_path)
            logDF.to_csv("save/trajectory_log.csv", index=False)
            break

        # Update vision.
        itemDF = objDetector(itemDF, controller, yolo_model)
        # Build and update action table.
        actionDF = actionTable(itemDF, conf_thres=0.5)
        actionDF = obstacleTable(actionDF, controller, threshold_distance=0.5)
        print("Updated Action Table:")
        print(actionDF)
        
        # Call GPT navigation to select an action.
        gpt_response = gptNav(controller, api_key, gpt_model, goal, actionDF, sourcePos)
        print("GPT response:", gpt_response)
        
        # Parse GPT response to extract the action id.
        action_id = parse_action_id(gpt_response)
        print("Parsed action id:", action_id)
        print("-----------")
        
        # Retrieve robot's current pose.
        agent_meta = controller.last_event.metadata["agent"]
        robot_x = agent_meta["position"].get("x", None)
        robot_y = agent_meta["position"].get("z", None)
        robot_yaw = agent_meta["rotation"].get("y", None)
        
        # Log the current step.
        log_entry = {
            "time": elapsed_time,
            "robot x": robot_x,
            "robot y": robot_y,
            "robot yaw": robot_yaw,
            "gpt decision": action_id
        }
        logDF = pd.concat([logDF, pd.DataFrame([log_entry])], ignore_index=True)
        
        # Map action id to a controller action.
        if action_id == 1:
            controller.step(action="MoveAhead")
            print("Executing action: Move Ahead.")
        elif action_id == 2:
            controller.step(action="RotateRight")
            print("Executing action: Rotate Right.")
        elif action_id == 3:
            controller.step(action="RotateLeft")
            print("Executing action: Rotate Left.")
        else:
            print("Invalid action id. Defaulting to Rotate Left.")
            controller.step(action="RotateLeft")
        
        time.sleep(1)

def parse_action_id(response_text):
    """
    Parses GPT response text to extract an action id (1, 2, or 3).
    Defaults to 3 (Rotate Left) if parsing fails.
    """
    for token in response_text.split():
        if token.isdigit():
            action_id = int(token)
            if action_id in [1, 2, 3]:
                return action_id
    return 3

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
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    api_key = config['OPENAI_KEY']
    gpt_model = config['OPENAI_CHAT_MODEL']
    
    itemColumns = ["name", "conf", "vizLoc", "glb3DLoc", "goalSimilarity", "searchPriority"]
    itemDF = pd.DataFrame(columns=itemColumns)
    
    goal = "food burning smell"
    target_items = ["Microwave"]
    
    yolo_model = YOLO("models/YOLO/yolov8s.pt")
    
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
    
    auto_control(
        controller=controller,
        itemDF=itemDF,
        yolo_model=yolo_model,
        api_key=api_key,
        gpt_model=gpt_model,
        target_items=target_items,
        save_path="save/itemDF.csv",
        max_time=150,
        goal=goal,
        dist_threshold=0.5
    )

if __name__ == "__main__":
    main()
