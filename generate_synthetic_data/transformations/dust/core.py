import cv2
import logging
import numpy as np

def execute(object_img, background_img, intensity=0.2, dust_opacity=0.4):
    """
    Add a dust effect to the object to match the dusty background.
    Args:
        object_img: The object image.
        background_img: The background image.
        intensity: Intensity of the dust noise (higher values = more dust).
        dust_opacity: Opacity of the dust overlay (0 = transparent, 1 = fully opaque).
    Returns:
        The object with the dust effect applied.
    """
    try:
        obj_h, obj_w = object_img.shape[:2]
        avg_dust_color = np.mean(background_img, axis=(0, 1)).astype(np.uint8)
        noise = np.random.normal(0, 255 * intensity, (obj_h, obj_w, 1)).astype(np.uint8)
        noise = np.repeat(noise, 3, axis=2)

        dust_overlay = np.full((obj_h, obj_w, 3), avg_dust_color, dtype=np.uint8)
        dust_overlay = cv2.add(dust_overlay, noise)

        dusty_object = cv2.addWeighted(object_img, 1 - dust_opacity, dust_overlay, dust_opacity, 0)
        return dusty_object
    
    except Exception as err:
        logging.info(f"Error adding dust effect: {err}")
        return object_img 
