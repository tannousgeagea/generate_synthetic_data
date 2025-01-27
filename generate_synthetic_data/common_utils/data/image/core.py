import cv2
import os

from common_utils.data.annotation.core import (
    load_yolo_segmentation_labels,
    load_yolo_boundingbox_labels,
    extract_polygon,
)

VALID_IMG_FORMAT = ['.png', '.jpg']

def load_images(src):
    return os.listdir(src)

def read_image(src, image_file):
    image_file = os.path.join(src, image_file)
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image not found: {image_file}")
    
    return cv2.imread(image_file)

def load_image_and_mask(image_path, annotation_dir, image_prefix="rgb", mask_prefix="panoptic", annotation_mode='mask'):
    """
    Load an image and its corresponding mask if it exists.
    
    Args:
        image_path (str): Path to the image file.
        mask_dir (str): Directory containing mask files.
        image_prefix (str): Prefix of the image filename (default: "rgb").
        mask_prefix (str): Prefix of the mask filename (default: "panoptic").
        mask_suffix (str): File extension for the mask files (default: ".png").
    
    Returns:
        tuple: Loaded image (np.ndarray), mask (np.ndarray or None).
    """
    # Extract the image filename without the prefix
    image_filename = os.path.basename(image_path)
    file_ext = f".{image_filename.split('.')[-1]}"
    core_filename = image_filename.replace(image_prefix, "").strip(file_ext) if annotation_mode=="mask" else image_filename.strip(file_ext)

    # Construct the corresponding mask filename
    annotation_filename = f"{mask_prefix}{core_filename}{file_ext}" if annotation_mode=='mask' else f"{core_filename}.txt"
    annotation_path = os.path.join(annotation_dir, annotation_filename)
    
    if not os.path.exists(annotation_path):
        raise FileExistsError(f'{annotation_path} does not exist')
    
    image = cv2.imread(image_path)
    if annotation_mode == 'mask':
        polygons, _ = extract_polygon(cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE), threshold_value=10 , epsilon_factor=0.01, debug=False)
    elif annotation_mode == 'seg':
        polygons = load_yolo_segmentation_labels(annotation_path, image.shape)
    elif annotation_mode == 'bbox':
        polygons = load_yolo_boundingbox_labels(annotation_path, image.shape)
    else:
        raise ValueError(f'undefined annotation mode: {annotation_mode}')
        
    return image, polygons
