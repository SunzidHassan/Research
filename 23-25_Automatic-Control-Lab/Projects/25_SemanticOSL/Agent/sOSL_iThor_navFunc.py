import numpy as np

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
