
import cv2
import random
import logging
import numpy as np

def execute(object_img, background_img, location, patch_size=20, num_patches=50, patch_opacity=0.5):
    """
    Add random-shaped patches from the background to the object for blending realism.
    Args:
        object_img: The object image.
        background_img: The background image.
        location: (x, y) top-left corner where the object is placed in the background.
        patch_size: Maximum size of the patches (in pixels).
        num_patches: Number of patches to overlay on the object.
        patch_opacity: Opacity of the background patches on the object.
    Returns:
        The object image with random-shaped background patches applied.
    """
    try:
        obj_h, obj_w = object_img.shape[:2]
        bg_h, bg_w = background_img.shape[:2]
        x, y = location

        # Ensure the object's location is within bounds
        x = max(0, min(x, bg_w - obj_w))
        y = max(0, min(y, bg_h - obj_h))

        # Copy the object image to avoid modifying the original
        patched_object = object_img.copy()

        for _ in range(num_patches):
            # Randomly sample a patch from the background near the object's location
            patch_x = random.randint(x, min(x + obj_w, bg_w) - patch_size)
            patch_y = random.randint(y, min(y + obj_h, bg_h) - patch_size)
            patch = background_img[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]

            # Generate a random mask with irregular shapes
            random_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
            num_points = random.randint(16, 24)  # Random number of vertices for the polygon
            points = np.array([
                [random.randint(0, patch_size), random.randint(0, patch_size)] for _ in range(num_points)
            ])
            cv2.fillPoly(random_mask, [points], 255)

            # Resize mask to match patch size and expand to 3 channels
            random_mask = cv2.resize(random_mask, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
            random_mask = np.expand_dims(random_mask, axis=-1)  # Shape (50, 50, 1)
            random_mask = np.repeat(random_mask, 3, axis=2)  # Shape (50, 50, 3)

            # Randomly choose a position on the object to place the patch
            obj_patch_x = random.randint(0, obj_w - patch_size)
            obj_patch_y = random.randint(0, obj_h - patch_size)

            # Convert patch and object to float32 for blending
            patch = patch.astype(np.float32)
            obj_area = patched_object[obj_patch_y:obj_patch_y + patch_size, obj_patch_x:obj_patch_x + patch_size].astype(np.float32)
            mask = (random_mask / 255).astype(np.float32)  # Normalize mask to range [0, 1]

            # Blend the patch onto the object using the random mask
            blended_area = obj_area * (1 - mask * patch_opacity) + patch * (mask * patch_opacity)

            # Write the blended area back to the object
            patched_object[obj_patch_y:obj_patch_y + patch_size, obj_patch_x:obj_patch_x + patch_size] = blended_area.astype(np.uint8)
            
    except Exception as err:
        logging.error(f"Error adding random patched: {err}")
        return object_img
    
    return patched_object