import numpy as np

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

# # Example usage:
# # Assume controller.last_event.metadata["objects"] holds the scene objects.
#   
# object_centers = get_objects_centers(objects, target_items)
# print("Centers of specified objects as a tensor:")
# print(object_centers)
