def obj_image_coordinate(results, object_name, names):
    """
    Extracts the (x, y) coordinates for all detections that match a given object name.

    Parameters:
        results: The YOLO detection results (assumed to be a list-like object where results[0].boxes
                 contains the detection boxes).
        object_name: The target object name as a string (e.g., "oven").
        names: A dictionary mapping class indices to class names (e.g., yolov8l.names).

    Returns:
        A list of tuples [(x, y), ...] for each detection matching the given object name.
        If no detection is found, returns an empty list.
    """
    coordinates = []
    
    # Loop over each detection box
    for box in results[0].boxes:
        # Each box.xywh[0] contains [x_center, y_center, width, height]
        x, y, w, h = box.xywh[0]
        # Get the predicted class index from box.cls[0] (convert to int)
        class_idx = int(box.cls[0].item())
        detected_name = names[class_idx]
        
        # Compare names (ignoring case)
        if detected_name.lower() == object_name.lower():
            coordinates.append((float(x), float(y)))
    
    return coordinates