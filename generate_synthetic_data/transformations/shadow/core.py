
import cv2
import logging
import numpy as np

def execute(new_image, mask, offset=(10, 10), intensity=0.5):
    """
    Add a shadow below the object.
    Args:
        new_image: Background image.
        mask: Object mask.
        offset: (x, y) shadow offset.
        intensity: Darkness level of the shadow (0 = transparent, 1 = fully dark).
    """
    try:
        # Shift the mask to create a shadow
        shadow_mask = np.zeros_like(mask)
        x_offset, y_offset = offset

        # Ensure offsets do not go out of bounds
        x_offset = min(x_offset, mask.shape[1] - 1)
        y_offset = min(y_offset, mask.shape[0] - 1)

        shadow_mask[y_offset:, x_offset:] = mask[:-y_offset, :-x_offset]

        # Resize shadow_mask to match new_image if necessary
        if shadow_mask.shape != new_image.shape[:2]:
            shadow_mask = cv2.resize(shadow_mask, (new_image.shape[1], new_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Apply the shadow to the background
        shadow_area = new_image.copy()
        shadow_indices = shadow_mask > 0
        shadow_area[shadow_indices] = (new_image[shadow_indices] * (1 - intensity)).astype(np.uint8)
    except Exception as err:
        logging.error(f"Error adding shadow: {err}")
        return new_image

    return shadow_area