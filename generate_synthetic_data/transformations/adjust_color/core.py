
import cv2
import logging
import numpy as np

def execute(object_img, background_img, location, alpha=0.5):
    """
    Subtly adjust the color of the object to blend with the background.
    Args:
        object_img: The object to adjust.
        background_img: The target background image.
        location: (x, y) position of the top-left corner of the object.
        alpha: Blending weight (0 = no adjustment, 1 = full adjustment to background color).
    """
    try:
        obj_h, obj_w = object_img.shape[:2]
        bg_region = background_img[location[1]:location[1]+obj_h, location[0]:location[0]+obj_w]

        # Calculate mean color of the object and the background region
        obj_mean = cv2.mean(object_img)[:3]
        bg_mean = cv2.mean(bg_region)[:3]

        # Subtly blend the object's mean color towards the background's mean color
        adjusted_object = object_img.astype(np.float32)
        for i in range(3):  # Iterate through BGR channels
            adjustment = alpha * (bg_mean[i] - obj_mean[i])  # Weighted adjustment
            adjusted_object[..., i] += adjustment

        # Clip values to valid range
        adjusted_object = np.clip(adjusted_object, 0, 255).astype(np.uint8)
    except Exception as err:
        logging.error(f"Error adjusting object color: {err}")
        return object_img

    return adjusted_object