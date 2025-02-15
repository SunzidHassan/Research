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

from ai2thor.controller import Controller
from ultralytics import YOLO
# from sentence_transformers import SentenceTransformer, util

# nav01
# from sOSL_iThor_reason import goalSimilarity, get_goal_location  # If available.
from sOSL_plumeField import get_field_value  # Ensure this is available
# from sOSL_iThor_objDetector import objDetector, globalLoc
from sOSL_iThor_navFunc import check_obstacle
from sOSL_iThor_gndTruth import get_objects_centers  # Imported helper

# nav02
from openai import OpenAI  # if needed
# llmPrompt is now integrated into gptNav

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

# ==========================
# SENSOR FUNCTIONS
# ==========================
def olfactionBranch(sourcePos, controller, plumeConcentration, q_s=2000, D=10, U=0, tau=1000, del_t=1, psi_deg=0):
    """
    Computes odor concentration based on the odor source position and the robot's current position.
    """
    robot_pos = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
    plumeConcentration = int(get_field_value(robot_pos, sources=sourcePos, q_s=q_s, D=D, U=U, tau=tau, del_t=del_t, psi_deg=psi_deg))
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

        # Estimate depth at the center of the bounding box.
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
    actionID = ['1', '2', '3']
    table = {"Action": actions, "ActionID": actionID, "Obstacle": ["" for _ in actions]}
    return pd.DataFrame(table)

def obstacleTable(actionDF, controller, obstacle_threshold):
    """
    Updates the Obstacle column in actionDF by analyzing the depth frame.
    Uses a central region to determine if there is an obstacle in front.
    """
    depth_frame = np.array(controller.last_event.depth_frame)
    forward_region = depth_frame[250:280,100:200]  # central region for forward.
    forward_min = np.min(forward_region)

    Left_region = depth_frame[250:280,0:50]  # central region for forward.
    Left_min = np.min(Left_region)
    Right_region = depth_frame[250:280,250:300]  # central region for forward.
    Right_min = np.min(Right_region)

    print(f'Forward min: {forward_min}, Left min: {Left_min}, Right min: {Right_min}, ')
    
    left_obstacle = "Yes" if Left_min < 0.5 else "No"
    forward_obstacle = "Yes" if forward_min < obstacle_threshold else "No"
    right_obstacle = "Yes"  if Right_min < 0.5 else "No"
    
    for idx, row in actionDF.iterrows():
        if row["Action"] == "Forward":
            actionDF.at[idx, "Obstacle"] = forward_obstacle
        elif row["Action"] == "Turn Left":
            actionDF.at[idx, "Obstacle"] = left_obstacle
        elif row["Action"] == "Turn Right":
            actionDF.at[idx, "Obstacle"] = right_obstacle
    
    return actionDF

# ==========================
# NEW HELPER FUNCTIONS
# ==========================
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
        "Authorization": f"Bearer {api_key}"}
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

def llmPrompt(delimiter, goal, actionDF_str, odor_concentration, prev_odor_concentration):
    """
    Generates a prompt for the GPT model to select one action from the action table.
    """
    task = f"""
    Your task is to select the best action for a mobile robot to move towards the source of {goal}.
    You are provided with an image an Action Table, and current and previous odor concentrations that summarizes the robot's current surroundings.
    
    The image includes:
    - Robot's current egocentric view.

    The table includes:
      - **Action**: The potential action (e.g., Move Forward, Rotate Left, Rotate Right).
      - **Obstacle**: Indicates if an obstacle is present ("Yes" or "No") in that direction.

    The odor concentration includes:
    - Integer value of current and past odor concentration.
    
    The rules are:
      1. If there are no obstacles in the Forward direction and the objects in the image align with the target odor source, the robot should move forward.
      2. If there is no object related to the {goal} present in any direction, the robot should turn left to explore a new view.
      3. If the Forward direction is blocked, then consider turning left or right if that direction is clear.
      4. Only one action should be selected.
      5. Always move towards high odor concentration area.
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

def gptNav(controller, api_key, gpt_model, goal, actionDF):
    delimiter = "#####"
    prompt = llmPrompt(delimiter, goal, actionDF.to_string(index=False))
    
    # Compress the current image.
    image_array = controller.last_event.frame
    compressed_bytes = compress_image(image_array, size=(64, 64))
    image_base64 = base64.b64encode(compressed_bytes).decode('utf-8')
    
    # Here we assume the API supports an "image" parameter; if not, you can alternatively
    # append the image to the prompt as a data URL.
    response = payload(api_key, prompt, image_base64, gpt_model)
    return response

# ==========================
# AUTOMATIC CONTROL LOOP
# ==========================
def auto_control(controller, itemDF, yolo_model, api_key, gpt_model, target_items, save_path="itemDF.csv", max_time=150, goal="food burning smell", dist_threshold=1.0, stepMagnitude=0.5):
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
        print("\n=============================")
        print("New Step")
        print("=============================\n")
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
        # itemDF = objDetector(itemDF, controller, yolo_model)
        # Build and update action table (only Action and Obstacle).
        actionDF = actionTable(itemDF, conf_thres=0.5)
        actionDF = obstacleTable(actionDF, controller, obstacle_threshold=1.0)
        print("Updated Action Table:")
        print(actionDF.to_string(index=False))
        
        # Call GPT navigation to select an action based solely on the (compressed) image.
        gpt_response = gptNav(controller, api_key, gpt_model, goal, actionDF)
        # print("GPT response:", gpt_response)
        
        # Parse GPT response to extract the action id.
        action_id = parse_action_id(gpt_response)
        print("\n")
        print("Parsed action id:", action_id)
        print("\n")
        # print("\n-----------\n")a
        
        # Retrieve robot's current pose.
        agent_meta = controller.last_event.metadata["agent"]
        robot_x = agent_meta["position"].get("x", None)
        robot_z = agent_meta["position"].get("z", None)  # using z for ground plane y-coordinate
        robot_yaw = agent_meta["rotation"].get("y", None)
        
        

        # Log the current step.
        log_entry = {
            "time": elapsed_time,
            "robot x": robot_x,
            "robot z": robot_z,
            "robot yaw": robot_yaw,
            "gpt decision": action_id
        }
        logDF = pd.concat([logDF, pd.DataFrame([log_entry])], ignore_index=True)
        
        # Map action id to a controller action.
        if action_id == 1:
            controller.step(action="MoveAhead", moveMagnitude=stepMagnitude)
            print("Executing action: Move Ahead.")
        elif action_id == 2:
            controller.step(action="RotateRight")
            controller.step(action="MoveAhead", moveMagnitude=0.1)
            
            print("Executing action: Rotate Right.")
        elif action_id == 3:
            controller.step(action="RotateLeft")
            controller.step(action="MoveAhead", moveMagnitude=0.1)
            print("Executing action: Rotate Left.")
        else:
            print("Invalid action id. Defaulting to Rotate Left.")
            controller.step(action="RotateLeft")
        
        print('\n')
        
        print(f"Robot x: {robot_x}, Robot z: {robot_z}")
        cv2.imshow("AI2-THOR", controller.last_event.cv2img)
        cv2.waitKey(int(1000))

        
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
    stepMagnitude = 0.5
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    api_key = config['OPENAI_KEY']
    gpt_model = config['OPENAI_CHAT_MODEL']
    
    itemColumns = ["name", "conf", "vizLoc", "glb3DLoc", "goalSimilarity", "searchPriority"]
    itemDF = pd.DataFrame(columns=itemColumns)
    
    goal = "food burning smell"
    target_items = ["Microwave"]
    
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
    
    # change the starting position
    i = 0
    
    actionID = [2,2,1,1,1,1,1]
    while i<len(actionID):
        if actionID[i] == 1:
            controller.step(action="MoveAhead", moveMagnitude=0.5)
            print("Executing action: Move Ahead.")
        elif actionID[i] == 2:
            controller.step(action="RotateRight")
            controller.step(action="MoveAhead", moveMagnitude=0.1)
            
            print("Executing action: Rotate Right.")
        elif actionID[i] == 3:
            controller.step(action="RotateLeft")
            controller.step(action="MoveAhead", moveMagnitude=0.1)
            print("Executing action: Rotate Left.")
        else:
            print("Invalid action id. Defaulting to Rotate Left.")
            controller.step(action="RotateLeft")
        i += 1

    auto_control(
        controller=controller,
        itemDF=itemDF,
        yolo_model=yolo_model,
        api_key=api_key,
        gpt_model=gpt_model,
        target_items=target_items,
        save_path="save/itemDF.csv",
        max_time=200,
        goal=goal,
        dist_threshold=1.5,
        stepMagnitude=stepMagnitude
    )

if __name__ == "__main__":
    main()
