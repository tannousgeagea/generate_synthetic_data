


def xywh2xywhn(xywh, image_shape):
    xmin, ymin, w, h = xywh
    return (
        (xmin + w / 2) / image_shape[1],
        (ymin + h / 2) / image_shape[2],
        w / image_shape[1],
        h / image_shape[0]
    )
    
def xyxy2xyxyn(xyxy, image_shape):
    """
    Convert bounding box coordinates from pixel format to normalized format.

    This function normalizes the bounding box coordinates based on the image dimensions. 
    The pixel coordinates (xmin, ymin, xmax, ymax) are converted to a normalized format 
    where each coordinate is represented as a fraction of the image's width or height.

    Parameters:
    - xyxy (tuple): A tuple of four integers (xmin, ymin, xmax, ymax) representing the bounding box coordinates in pixel format.
    - image_shape (tuple): A tuple of two integers (height, width) representing the dimensions of the image.

    Returns:
    - tuple: A tuple of four floats (xmin_n, ymin_n, xmax_n, ymax_n) representing the normalized bounding box coordinates.
    """
    xmin, ymin, xmax, ymax = xyxy
    return (xmin / image_shape[1], ymin / image_shape[0], xmax / image_shape[1], ymax / image_shape[0])

def xyxy2xywh(xyxy):
    """
    Convert bounding box coordinates from (xmin, ymin, xmax, ymax) format to (x, y, width, height) format.

    Parameters:
    - xyxy (Tuple[int, int, int, int]): A tuple representing the bounding box coordinates in (xmin, ymin, xmax, ymax) format.

    Returns:
    - Tuple[int, int, int, int]: A tuple representing the bounding box in (x, y, width, height) format. 
                                 (x, y) are  the center of the bounding box.
    """
    xmin, ymin, xmax, ymax = xyxy
    w = xmax - xmin
    h = ymax - ymin
    return (xmin + w/2, ymin + h/2, w, h)