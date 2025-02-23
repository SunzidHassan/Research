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
# from sentence_transformers import SentenceTransformer, util

# nav01
# from sOSL_iThor_reason import goalSimilarity, get_goal_location  # If available.
# from sOSL_plumeField import get_field_value  # Ensure this is available
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
    actions = ["Forward", "Turn Right", "Turn Left", "Turn Back"]
    actionID = ['1', '2', '3', '4']
    table = {"Action": actions, "ActionID": actionID, "Obstacle": ["" for _ in actions]}
    return pd.DataFrame(table)

def obstacleTable(actionDF, controller, obstacle_threshold):
    """
    Updates the Obstacle column in actionDF by analyzing the depth frame.
    Uses a central region to determine if there is an obstacle in front.
    """
    depth_frame = np.array(controller.last_event.depth_frame)
    forward_region = depth_frame[280:300,100:200]  # central region for forward.
    forward_min = np.min(forward_region)

    Left_region = depth_frame[250:280,0:50]  # central region for forward.
    Left_min = np.min(Left_region)
    Right_region = depth_frame[250:280,250:300]  # central region for forward.
    Right_min = np.min(Right_region)

    print(f'Forward min: {forward_min}, Left min: {Left_min}, Right min: {Right_min}, \n')
    print(f'Obstacle Threshold: {obstacle_threshold}\n')
    
    left_obstacle = "Yes" if Left_min < 0.3 else "No"
    forward_obstacle = "Yes" if forward_min < obstacle_threshold else "No"
    right_obstacle = "Yes"  if Right_min < 0.3 else "No"
    back_obstacle = "No"
    
    for idx, row in actionDF.iterrows():
        if row["Action"] == "Forward":
            actionDF.at[idx, "Obstacle"] = forward_obstacle
        elif row["Action"] == "Turn Left":
            actionDF.at[idx, "Obstacle"] = left_obstacle
        elif row["Action"] == "Turn Right":
            actionDF.at[idx, "Obstacle"] = right_obstacle
        elif row["Action"] == "Turn Back":
            actionDF.at[idx, "Obstacle"] = back_obstacle
    
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
def payload(api_key, prompt, image, model="gpt-4o"):
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()['choices'][0]['message']['content']
    return response

def llmPrompt(delimiter, goal, actionDF_str, last_action, odor_concentration, prev_odor_concentration=0, step_count=0):
    
    """
    Generates a prompt for the GPT model to select one action from the action table.
    """
    task = f"""
    Your task is to select the best action for a mobile robot to move towards the source of {goal}.
    You are provided with an image, an Action Table, the last selected action, and current and previous odor concentrations that summarizes the robot's current surroundings.
    
    The image includes:
    - Robot's current egocentric view.

    The table includes:
      - **Action**: The potential action (e.g., Move Forward, Rotate Left, Rotate Right).
      - **Obstacle**: Indicates if an obstacle is present ("Yes" or "No") in that direction.

    The last action includes:
    - The Action_id of the last action taken by the robot.

    The odor concentration includes:
    - Value of current and past odor concentration.
    """
    
    rlues_old = f"""
    The rules are listed acoording to priority:
      1. Move forward if there is no obstacle in front.
      2. If the scene may contain an object related to the {goal}, the robot should move forward.
      3. If odor concentration decreases after executing moveAhead, then Turn back.
      4. If the forward direction is blocked, then turn to the side (Turn Left or Turn Right) that is obstacle free and the {goal} related object may be located.
      5. If the last action was to turn (Action_id = 2 or 3), and the forward direction is blocked, then Turn Back.
      6. Only one action should be selected.
    """

    guides = f"""
      * Only one action should be selected.
    """
    
    semanticAnalysis = f"""
    Sequentially perform the following input analysis to select the best action:
    * Visual Analysis: do you see an object that is a possible source for {goal}?
        * If yes, list the most likely object's name.
        * If yes, and the front is obstacle free, move forward.
    * Olfactory Analysis: 
        * If the sensed concentration decreases, turn back.
        * If the sensed concentration increases and front is obstacle free, move forward. 
        * If the sensed concentration stays the same, focus on visual and navigation analysis.
    * Navigation Analysis:
        * If the front is blocked, turn to the side that is obstacle free and is more likely to lead you to the possible {goal} related object.
        * If the last action was to turn (Action_id = 2 or 3), and the forward direction is blocked, then Turn Back.
    """
    
    actionInstructions = """
    Move forward. (Action_id = 1)
    Turn right. (Action_id = 2)
    Turn left. (Action_id = 3)
    Turn back. (Action_id = 4)
    """
    
    reasoningOutputInstructions = f"""
    Your response should follow this format:
    {delimiter} 
    Visual Analysis:
    <Visual analysis reasoning>
    ...
    {delimiter} 
    Olfactiory Analysis:
    <olfactory analysis reasoning>
    ...
    {delimiter}
    Navigation Analysis:
    <navigation analysis reasoning>
    ...
    {delimiter}
    Selected Action:
    <Output only one Action_id as an integer>
    """

    noReasoningOutputInstructions = f"""
    Respond with the corresponding numerical value of the action (1, 2, 3) without any additional text or punctuation.
    """
    
    LastAction = f"Action_id at the previous time step was: {last_action}"
    
    olfactoryReading = f"""
    Odor concentration at the previous time step was: {prev_odor_concentration}
    Odor Concentration at the current time step is: {odor_concentration}
    """
    
    prompt = f"""
    {delimiter} 
    Task:
    {task}
    {delimiter}
    Olfactory reading:
    {olfactoryReading}
    {delimiter}
    Current Action Table:
    {actionDF_str}
    {delimiter}
    Available Actions:
    {actionInstructions}
    {delimiter}
    Last Action:
    {LastAction}
    {delimiter}
    Step-by-step Analysis:
    {semanticAnalysis}
    {delimiter}
    Output Reasoning:
    {reasoningOutputInstructions}
    """

    
    # if step_count == 0:
    #     prompt = f"""
    #     {delimiter} Task:
    #     {task}
    #     {delimiter} Available Actions:
    #     {actionInstructions}
    #     {delimiter} Current Action Table:
    #     {actionDF_str}
    #     {delimiter} Output Instructions:
    #     {reasoningOutputInstructions}
    #     {delimiter} Olfactory reading:
    #     {olfactoryReading}
    #     """
    # else:
    #     prompt = f"""
    #     {delimiter} Olfactory reading:
    #     {olfactoryReading}
    #     """
    
    return prompt


def gptNav(controller, api_key, gpt_model, goal, actionDF, source_position, step_count, last_action):
    # Retrieve previous odor concentration from the function attribute, if it exists.
    prev_odor_concentration = getattr(gptNav, "prev_odor_concentration", None)
    
    # Get the current odor concentration.
    current_odor_concentration = olfactionBranch(source_position, controller)
    
    # If no previous value exists, initialize it.
    if prev_odor_concentration is None:
        prev_odor_concentration = current_odor_concentration

    print(f"Prev Odor Concentration: {prev_odor_concentration}")
    print(f"Current Odor Concentration: {current_odor_concentration}\n")
    
    delimiter = "#####"
    prompt = llmPrompt(delimiter, goal, actionDF.to_string(index=False),
                        last_action, current_odor_concentration, prev_odor_concentration, step_count)
    
    # Compress the current image.
    image_array = controller.last_event.frame
    compressed_bytes = compress_image(image_array, size=(64, 64))
    image_base64 = base64.b64encode(compressed_bytes).decode('utf-8')
    
    response = payload(api_key, prompt, image_base64, gpt_model)
    
    # Save the current odor concentration as the previous one for the next call.
    gptNav.prev_odor_concentration = current_odor_concentration
    
    return response

# ==========================
# AUTOMATIC CONTROL LOOP
# ==========================

def auto_control(controller, itemDF, yolo_model, api_key, gpt_model, source_position, 
                 save_path="itemDF.csv", max_time=150, goal="", 
                 dist_threshold=1.0, stepMagnitude=0.5):
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
      6. Logs the time, robot position (x, z), robot yaw, and GPT decision.
    """
    
    step_count = 1    
    start_time = time.time()
    logDF = pd.DataFrame(columns=["step", "robot x", "robot z", "robot yaw", 
                                  "gpt decision", "concentration", "front obstacle", "reasoning"])
    
    print("Automatic control active. Executing GPT-selected actions until timeout or target reached.")
    
    while True:
        print("\n=============================")
        print("New Step")
        print("=============================\n")
        elapsed_time = time.time() - start_time
        print(f"Steps: {step_count}/40")
        
        # Evaluate sensor readings: compute minimum distance.
        if source_position.size > 0:
            distances = [get_distance_to_source(controller, center) for center in source_position]
            min_distance = min(distances)
        else:
            min_distance = float('inf')
        
        print(f"Current minimum distance to target: {min_distance:.2f}")
        
        # get current odor concentration
        currentConcentration = olfactionBranch(source_position, controller)
        
        # Variable obstacle threshold
        obstacle_threshold = np.interp(currentConcentration, [0, 0.3], [1, 0.6])
        
        # Build and update action table.
        actionDF = actionTable(itemDF, conf_thres=0.5)
        actionDF = obstacleTable(actionDF, controller, obstacle_threshold)
        print("Updated Action Table:")
        print(actionDF.to_string(index=False))
        
        # Pass last action if available (for GPT context).
        last_action = getattr(auto_control, "last_action", None)
        
        # Call GPT navigation to select an action.
        gpt_response = gptNav(controller, api_key, gpt_model, goal, actionDF, 
                               source_position, step_count, last_action)
        print("GPT response:", gpt_response)
        
        # Parse the GPT response.
        action_id = parse_action_id(gpt_response)
        print("\nParsed action id:", action_id, "\n")
        auto_control.last_action = action_id
        
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
            "gpt decision": action_id,
            "concentration": currentConcentration,
            "front obstacle": actionDF[actionDF["Action"] == "Forward"]["Obstacle"].values[0],
            "reasoning": gpt_response
        }
        logDF = pd.concat([logDF, pd.DataFrame([log_entry], columns=logDF.columns)], 
                          ignore_index=True)
        
        stepMagnitude = np.interp(currentConcentration, [0, 0.3], [0.7, 0.25])
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
        elif action_id == 4:
            controller.step(action="RotateLeft", degrees=180)
            controller.step(action="MoveAhead", moveMagnitude=0.1)
            print("Executing action: Turn Back.")
        else:
            print("Invalid action id. Defaulting to Rotate Left.")
            controller.step(action="RotateLeft")
        
        # Save the current vision frame.
        frame_filename = f"save/{step_count}.png"
        cv2.imwrite(frame_filename, controller.last_event.cv2img)
        print(f"Saved vision frame as {frame_filename}")
        
        print(f"Robot x: {robot_x}, Robot z: {robot_z}")
        cv2.imshow("AI2-THOR", controller.last_event.cv2img)
        cv2.waitKey(int(1000))
        
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



def parse_action_id(response_text):
    """
    Parses GPT response text to extract an action id (1, 2, or 3) from the text following
    the "Selected Action:" marker, regardless of additional delimiter characters.
    It finds the first integer after the marker and returns it if it's valid.
    Defaults to 3 if parsing fails.
    """
    marker = "Selected Action:"
    idx = response_text.find(marker)
    if idx != -1:
        # Extract text after the marker
        after_marker = response_text[idx + len(marker):]
        # Look for the first integer in the text after the marker
        match = re.search(r"\d+", after_marker)
        if match:
            action_id = int(match.group())
            if action_id in [1, 2, 3, 4]:
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
    stepMagnitude = 0.25
    
    
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    api_key = config['OPENAI_KEY']
    gpt_model = config['OPENAI_CHAT_MODEL']
    
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
    
    goal = "rotten smell"
    target_items = ["GarbageCan"]
    
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


    # Garbage Start Pos 3:
    controller.step(
        action="Teleport",
        position=dict(x=-1, y=0.9, z=-1.5),
        rotation=dict(x=0, y=90, z=0),
    )

    controller.step(
        "MoveAhead",
        moveMagnitude=0.01
    )

    auto_control(
        controller=controller,
        itemDF=itemDF,
        yolo_model=yolo_model,
        api_key=api_key,
        gpt_model=gpt_model,
        source_position=sourcePos,
        save_path="save/itemDF.csv",
        max_time=200,
        goal=goal,
        dist_threshold=0.8,
        stepMagnitude=stepMagnitude
    )

if __name__ == "__main__":
    main()
