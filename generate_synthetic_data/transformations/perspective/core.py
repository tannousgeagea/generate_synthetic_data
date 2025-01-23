import cv2
import logging
import numpy as np

def execute(object_img, mask, src_points, dst_points):
    """
    Apply perspective transformation to the object and mask.
    """
    try:
        matrix = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
        warped_object = cv2.warpPerspective(object_img, matrix, (object_img.shape[1], object_img.shape[0]))
        warped_mask = cv2.warpPerspective(mask, matrix, (mask.shape[1], mask.shape[0]))
    except Exception as err:
        logging.error(f"Error applying perspective transformation: {err}")
        return object_img, mask
    
    return warped_object, warped_mask