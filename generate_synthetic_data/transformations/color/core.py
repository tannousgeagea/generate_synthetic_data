import cv2
import logging

def execute(object_img, mask, hue_shift):
    """
    Change the hue of the object while keeping other properties intact.
    Args:
        object_img: The object image (BGR format).
        mask: Binary mask of the object.
        hue_shift: Amount to shift the hue (0-179 in OpenCV's HSV space).
    Returns:
        Color-modified object image.
    """
    
    result = object_img
    try:
        # Convert the object image to HSV color space
        hsv = cv2.cvtColor(object_img, cv2.COLOR_BGR2HSV)

        # Apply the mask to isolate the object
        object_hsv = hsv.copy()
        object_hsv[mask == 0] = 0  # Zero out non-object areas

        # Shift the hue channel
        object_hsv[..., 0] = (object_hsv[..., 0] + hue_shift) % 180

        # Convert back to BGR color space
        modified_object = cv2.cvtColor(object_hsv, cv2.COLOR_HSV2BGR)

        # Combine with the original image to retain non-object areas
        result = cv2.bitwise_and(modified_object, modified_object, mask=mask) + \
                cv2.bitwise_and(object_img, object_img, mask=cv2.bitwise_not(mask))

    except Exception as err:
        logging.info(f"Error changing object hue: {err}")

    return result