import logging
import numpy as np
from skimage.exposure import match_histograms

def execute(object_img, background_img, location):
    """
    Match the histogram of the object to the background region.
    """
    try:
        obj_h, obj_w = object_img.shape[:2]
        bg_region = background_img[location[1]:location[1]+obj_h, location[0]:location[0]+obj_w]

        # Ensure the object and background region have the same number of channels
        if object_img.shape[-1] == 3 and bg_region.shape[-1] == 3:  # Both are RGB
            matched_object = match_histograms(object_img, bg_region)
        else:
            return object_img
        
    except Exception as err:
        logging.error(f"Error matching color histogram: {err}")
        return object_img

    return matched_object.astype(np.uint8)