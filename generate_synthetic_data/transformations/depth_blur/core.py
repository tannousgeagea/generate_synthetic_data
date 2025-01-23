import cv2
import logging

def execute(object_img, distance, focus_distance=7, max_blur=10):
    """
    Blur the object based on its distance from the camera.
    Args:
        object_img: The object to blur.
        distance: The distance of the object (in meters).
        focus_distance: The distance in focus.
        max_blur: Maximum blur amount for distant objects.
    """
    
    try:
        blur_amount = int(max(0, min(max_blur, (distance - focus_distance) / focus_distance * max_blur)))
        if blur_amount > 0:
            blurred_object = cv2.GaussianBlur(object_img, (2 * blur_amount + 1, 2 * blur_amount + 1), 0)
            return blurred_object
    except Exception as err:
        logging.info(f'Error applying depth blur: {err}')
        return object_img
    
    return object_img