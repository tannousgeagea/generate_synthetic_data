import cv2
import math
import logging

def calculate_apparent_size_fov(object_real_height, depth, fov, image_height):
    """
    Calculate the apparent size of the object in the image based on the field of view (FOV).
    Args:
        object_real_height: The real-world height of the object (in meters).
        depth: The distance of the object from the camera (in meters).
        fov: The vertical field of view of the camera (in degrees).
        image_height: The height of the image (in pixels).
    Returns:
        Apparent height of the object in pixels.
    """
    # Convert FOV from degrees to radians
    fov_rad = math.radians(fov)

    # Calculate the apparent size in pixels
    apparent_size_in_image = (object_real_height * image_height) / (2 * depth * math.tan(fov_rad / 2))
    return int(apparent_size_in_image)

def execute(object_img, mask, object_real_height, depth, fov, image_height):
    """
    Resize the object and mask to match its apparent size at a given depth using FOV.
    """
    try:
        # Calculate the apparent size in pixels
        apparent_height = calculate_apparent_size_fov(object_real_height, depth, fov, image_height)

        # Get the scaling factor
        scaling_factor = apparent_height / object_img.shape[0]

        # Resize the object and mask
        new_width = int(object_img.shape[1] * scaling_factor)
        new_height = apparent_height
        resized_object = cv2.resize(object_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_AREA)

    except Exception as err:
        logging.error(f"Error resizing object to depth fov: {err}")
        return object_img, mask

    return resized_object, resized_mask

def crop_image_to_center(image, mask, crop_width, crop_height):
    """
    Crop the object image and mask to the center.
    Args:
        image (np.ndarray): The input object image.
        mask (np.ndarray): The corresponding binary mask.
        crop_width (int): Width of the crop area.
        crop_height (int): Height of the crop area.
    Returns:
        tuple: Cropped image and mask.
    """
    height, width = image.shape[:2]

    # Ensure the crop dimensions do not exceed the original dimensions
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)

    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Calculate the cropping box
    x1 = max(0, center_x - crop_width // 2)
    y1 = max(0, center_y - crop_height // 2)
    x2 = min(width, center_x + crop_width // 2)
    y2 = min(height, center_y + crop_height // 2)

    # Crop the image and mask
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2] if mask is not None else None

    return cropped_image, cropped_mask

def execute_with_crop(object_img, mask, object_real_height, depth, fov, image_height):
    """
    Crop the object and mask to match the apparent size at a given depth using FOV.
    Args:
        object_img (np.ndarray): The object image.
        mask (np.ndarray): The binary mask of the object.
        object_real_height (float): The real-world height of the object in meters.
        depth (float): The distance of the object from the camera in meters.
        fov (float): The field of view of the camera in degrees.
        image_height (int): The height of the image in pixels.
    Returns:
        tuple: Cropped object image and mask.
    """
    try:
        # Calculate the apparent size in pixels
        apparent_height = calculate_apparent_size_fov(object_real_height, depth, fov, image_height)

        # Calculate the cropping dimensions based on the apparent height
        crop_width = int(object_img.shape[1] * (apparent_height / object_img.shape[0]))
        crop_height = apparent_height

        # Crop the image and mask
        cropped_object, cropped_mask = crop_image_to_center(object_img, mask, crop_width, crop_height)

    except Exception as err:
        logging.error(f"Error cropping object to depth FOV: {err}")
        return object_img, mask

    return cropped_object, cropped_mask
