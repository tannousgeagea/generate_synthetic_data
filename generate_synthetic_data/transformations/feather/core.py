import cv2
import logging

def execute(mask, kernel_size=21):
    """
    Feather the edges of the mask to create smoother transitions.
    """
    try:
        blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    
    except Exception as err:
        logging.error(f"Error feathering the edges: {err}")
        mask
        
    return blurred_mask