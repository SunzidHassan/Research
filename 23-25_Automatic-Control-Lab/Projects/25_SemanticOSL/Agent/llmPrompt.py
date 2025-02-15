def llmPrompt(delimiter, goal, actionDF_str, plume):
    """
    Generates a prompt for the GPT model to select one action from the action table.
    
    Parameters:
      delimiter (str): A string delimiter used to separate sections.
      goal (str): The target odor source (e.g., "food burning smell").
      actionDF_str (str): A string representation of the action table with columns "Action", "Obstacle", and "Objects".
      plume (str): A string representing chemical concentration value.
    
    Returns:
      str: The generated prompt.
    """
    task = f"""
    Your task is to select the best action for a mobile robot to move towards the source of {goal}.
    You are provided with an Action Table that summarizes the robot's current surroundings.
    
    The table includes:
      - **Action**: The potential action (e.g., Move Forward, Rotate Left, Rotate Right).
      - **Obstacle**: Indicates if an obstacle is present ("Yes" or "No") in that direction.
      - **Objects**: Lists objects detected in that direction.
    
    The rules are:
      1. If there are no obstacles in the Forward direction and the objects in that row align with the target odor source, the robot should move forward.
      2. If there is no object related to the {goal} present in any direction, the robot should turn left to explore a new view.
      3. If the Forward direction is blocked, then consider turning left or right if that direction is clear and the detected objects align with the {goal}.
      4. Only one action should be selected.
      5. If chemical concentration increases as the robot moves towards the source, the robot should move in that direction.
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