import numpy as np
import cv2

def sample_texture_from_background(background_img, patch_size=50, num_patches=5):
    """
    Sample random texture patches from the background.
    Args:
        background_img (np.ndarray): The background image.
        patch_size (int): Size of the texture patch.
        num_patches (int): Number of texture patches to extract.
    Returns:
        list: List of texture patches.
    """
    height, width = background_img.shape[:2]
    texture_patches = []

    for _ in range(num_patches):
        x = np.random.randint(0, width - patch_size)
        y = np.random.randint(0, height - patch_size)
        patch = background_img[y:y + patch_size, x:x + patch_size]
        texture_patches.append(patch)

    return texture_patches

def blend_textures(object_img, mask, textures, opacity=0.3):
    """
    Blend texture patches onto the object image.
    Args:
        object_img (np.ndarray): The object image.
        mask (np.ndarray): Binary mask of the object.
        textures (list): List of texture patches.
        opacity (float): Blending opacity (0 to 1).
    Returns:
        np.ndarray: Object image with blended textures.
    """
    obj_h, obj_w = object_img.shape[:2]
    blended_image = object_img.copy()

    for texture in textures:
        # Resize texture to match a random region of the object
        tex_h, tex_w = np.random.randint(20, 50, size=2)
        resized_texture = cv2.resize(texture, (tex_w, tex_h), interpolation=cv2.INTER_AREA)

        # Select a random location on the object
        x = np.random.randint(0, obj_w - tex_w)
        y = np.random.randint(0, obj_h - tex_h)

        # Extract the region of the object and blend the texture
        roi = blended_image[y:y + tex_h, x:x + tex_w]
        texture_mask = mask[y:y + tex_h, x:x + tex_w]

        if roi.shape == resized_texture.shape:
            blended_texture = cv2.addWeighted(roi, 1 - opacity, resized_texture, opacity, 0)
            # Apply the blended texture only within the mask
            blended_image[y:y + tex_h, x:x + tex_w] = np.where(
                texture_mask[..., None] > 0, blended_texture, roi
            )

    return blended_image
