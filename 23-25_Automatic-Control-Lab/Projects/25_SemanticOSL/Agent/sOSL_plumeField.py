import math
import numpy as np

def get_field_value(robot_pos, sourcesPos, q_s=2000, D=10, U=0, tau=1000, del_t=1, psi_deg=0):
    """
    Computes the odor field value at a given robot position as the sum of contributions
    from one or more odor sourcesPos.
    
    Parameters:
        robot_pos  (ndarray): A NumPy array of shape (2,) representing the robot's [x, y] position.
        sourcesPos    (ndarray): A NumPy array of shape (n,2) where each row is [x_s, y_s] for a source.
        q_s        (float): Source strength.
        D          (float): Diffusion coefficient.
        U          (float): Advection velocity (set to 0 if no airflow).
        tau        (float): Time or scaling parameter.
        del_t      (float): Time step.
        psi_deg    (float): Angle in degrees for rotation (direction of advection; irrelevant if U=0).
    
    Returns:
        (float): The computed field value at the robot's position as the sum of contributions
                 from all sourcesPos.
    """
    # Convert psi from degrees to radians
    psi = math.radians(psi_deg)
    
    # Compute lambda; note that if U==0, lambda simplifies to sqrt(D*tau)
    lambd = math.sqrt((D * tau) / (1 + (tau * U**2) / (4 * D)))
    
    total = 0.0
    x, y = robot_pos  # Unpack the robot position
    
    # Loop over each source
    for source in sourcesPos:
        x_s, y_s = source  # Unpack the source coordinates
        
        # Compute differences relative to the odor source
        delta_x, delta_y = x - x_s, y - y_s
        
        # Compute Euclidean distance from the point to the source
        r = np.hypot(delta_x, delta_y)
        
        # Avoid division by zero if r==0
        if r == 0:
            contribution = - (r / lambd) * del_t
        else:
            # Compute the rotated y coordinate (irrelevant if U==0)
            rotated_y = -delta_x * math.sin(psi) + delta_y * math.cos(psi)
            
            # First term: advection component (exponential becomes 1 if U==0)
            term1 = (q_s / (4 * math.pi * D * r)) * math.exp(-rotated_y * U / (2 * D))
            
            # Second term: diffusion component
            term2 = - (r / lambd) * del_t
            
            contribution = term1 + term2
        
        total += contribution
        
    return total
