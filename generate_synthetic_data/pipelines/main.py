

import os
import random
import  logging
import numpy as np
import cv2
import shutil
from tqdm import tqdm

from common_utils.data.annotation.core import (
    load_yolo_segmentation_labels,
    extract_polygon,
    extract_object,
    get_bounding_box_from_mask,
    draw_polygon_bounding_box_from_mask,
)


from common_utils.data.annotation.convertor import (
    xyxy2xyxyn,
    xyxy2xywh,
)

from common_utils.data.image.core import (
    load_image_and_mask
)

from transformations import (
    color as color_fn,
    feather as feather_fn,
    shadow as shadow_fn,
    adjust_color,
    depth_blur as depth_blur_fn,
    dust as dust_fn,
    histogram_match as histogram_match_fn,
    perspective as perspective_fn,
    random_patches,
    blend,
    resize,
    blend_texture,
)


SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
VALID_IMG_FORMAT = ['.png', '.jpg']

def apply_transformations(object_img, mask):
    """
    Apply random transformations (rotation and shear) to the object.
    """
    rows, cols = object_img.shape[:2]
    center = (cols // 2, rows // 2)

    # Random rotation
    angle = random.uniform(-20, 20)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # Random shear
    shear_factor = random.uniform(-0.2, 0.2)
    shear_matrix = np.array([[1, shear_factor, 0], [shear_factor, 1, 0]], dtype=np.float32)

    rotated = cv2.warpAffine(object_img, rotation_matrix, (cols, rows))
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (cols, rows))

    sheared = cv2.warpAffine(rotated, shear_matrix, (cols, rows))
    sheared_mask = cv2.warpAffine(rotated_mask, shear_matrix, (cols, rows))
    
    return sheared, sheared_mask

def insert_object_randomly(
    new_image, 
    object_img,
    mask,
    location,
    center,
    d_orig,
    d_new,
    perspective=False,
    patches=True,
    color=True,
    dust=False,
    shadow=True,
    feather=True,
    depth_blur=True,
    is_texture=True,
    histogram_match=True
    ):
    """
    Insert the object at a random location with advanced blending.
    """
    try:
        obj_h, obj_w = object_img.shape[:2]

        if color:
            hue_shift = random.randint(0, 30)
            object_img = color_fn.core.execute(object_img, mask, hue_shift=hue_shift)
            
        # Feather the edges of the mask
        if feather:
            mask = feather_fn.core.execute(mask)
        
        if shadow:
            new_image = shadow_fn.core.execute(new_image, mask, offset=(3, 3), intensity=0.05)
        
        if perspective:
            src_points = np.float32([[0, 0], [obj_w - 1, 0], [obj_w - 1, obj_h - 1], [0, obj_h - 1]])
            dst_points = np.float32([[10, 0], [obj_w - 10, 0], [obj_w - 20, obj_h], [20, obj_h]])
            object_img, mask = perspective_fn.core.execute(object_img, mask, src_points, dst_points)

        if depth_blur:
            object_img = depth_blur_fn.core.execute(object_img, d_orig, d_new)

        if histogram_match:
            object_img = histogram_match_fn.core.execute(object_img, new_image, location)

        if dust:
            object_img = dust_fn.core.execute(object_img, new_image, intensity=0.01, dust_opacity=0.1)

        if patches:
            object_img = random_patches.core.execute(object_img, new_image, location, patch_size=250, num_patches=2, patch_opacity=0.5)
        
        
        if is_texture:
            textures = blend_texture.core.sample_texture_from_background(new_image, patch_size=50, num_patches=5)
            object_img = blend_texture.core.blend_textures(object_img, mask, textures, opacity=0.3)
        
        object_img = adjust_color.core.execute(object_img, new_image, location, alpha=0.3)
        new_image = blend.core.execute(new_image, object_img, mask, center)

    except Exception as err:
        logging.error(f"Error inserting random objects: {err}")

    return new_image

def execute(
    image, polygons, backgound, depth=7.5, fov=10, object_real_height=2, debug=False
):
    for polygon in polygons:
        object_img, mask = extract_object(image, polygon)
        transformed_obj, transformed_mask = apply_transformations(object_img, mask)
        resized_object, resized_mask = resize.core.execute_with_crop(
            transformed_obj, transformed_mask, object_real_height=object_real_height, depth=depth, fov=fov, image_height=image.shape[0],
        )

        location = (
            random.randint(0, backgound.shape[1] - resized_object.shape[1]),
            random.randint(0, backgound.shape[0] - resized_object.shape[0])
            )
        
        center = (
            location[0] + resized_object.shape[1] // 2, 
            location[1] + resized_object.shape[0] // 2
            )
        
        new_image = insert_object_randomly(
            backgound,
            resized_object,
            resized_mask,
            location,
            center,
            30,
            5,
            perspective=True,
            shadow=True,
            feather=True,
            depth_blur=False,
            dust=True,
            histogram_match=False,
            patches=False,
        )
        
        bounding_box = get_bounding_box_from_mask(mask, center)
        if debug:
            draw_polygon_bounding_box_from_mask(
                new_image, 
                bounding_box, 
                color=(0, 255, 255), 
                thickness=2
                )
            
    return new_image, bounding_box
        

ANNOTAION = [
    'mask', 'txtfile'
]

def get_all_files(directory, extensions=[".jpg", ".png"]):
    """
    Get all files in a directory with specified extensions.
    Args:
        directory (str): Path to the directory.
        extensions (list): List of file extensions to include.
    Returns:
        list: List of file paths.
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files

def load_random_object(objects_dir):
    """
    Load a random object category and a random orientation for that object.
    Args:
        objects_dir (str): Path to the objects directory.
    Returns:
        tuple: (object_image, object_mask, object_name)
    """
    # Get all object categories
    object_categories = [d for d in os.listdir(objects_dir) if os.path.isdir(os.path.join(objects_dir, d))]
    
    # Randomly select an object category
    selected_category = random.choice(object_categories)
    category_path = os.path.join(objects_dir, selected_category)
    
    images_path = os.path.join(category_path, "images")
    masks_path = os.path.join(category_path, "masks")
    
    images = get_all_files(images_path)
    image_path = random.choice(images)
    
    image, polygons = load_image_and_mask(
        image_path=image_path,
        annotation_dir=masks_path,
        image_prefix='rgb',
        mask_prefix='panoptic',
        annotation_mode='mask',
    )
    
    return image, polygons, image_path, selected_category

def process_background(background_image, objects_dir, output_dir, max_objects=5):
    """
    Process a single background image by blending random objects into it.
    Args:
        background_image (str): Path to the background image.
        objects_dir (str): Path to the objects directory.
        output_dir (str): Directory to save the blended image.
        max_objects (int): Maximum number of objects to blend into the background.
    """
    # Load the background image
    background = cv2.imread(background_image)
    image_file = os.path.basename(background_image)
    num_objects = random.randint(2, max_objects)
    for _ in range(num_objects):
        image, polygons, image_path, category = load_random_object(objects_dir)
        new_image, xyxy = execute(
                image,
                polygons,
                background,
            )
        
        new_image_file = f"{image_file}_{category}_{os.path.basename(image_path)}"
        file_ext = f".{new_image_file.split('.')[-1]}"
        txt_file = f"{new_image_file.strip(file_ext)}.txt"
        
        xywh = xyxy2xywh(
            xyxy=xyxy2xyxyn(xyxy, image_shape=new_image.shape)
        )    
        
        data = [
            [0] + list(xywh)
        ]
        
        lines = (("%g " * len(line)).rstrip() % tuple(line) + "\n" for line in data)
        
        image_location = os.path.join(output_dir, 'images')
        label_location = os.path.join(output_dir, "labels")
        os.makedirs(image_location, exist_ok=True)
        os.makedirs(label_location, exist_ok=True)
        
        cv2.imwrite(os.path.join(image_location, new_image_file), new_image)
        with open(label_location + "/" + txt_file, "w") as file:
            file.writelines(lines)

def main_pipeline(background_dir, objects_dir, output_dir, max_objects=5):
    """
    Main pipeline to blend objects into backgrounds.
    Args:
        background_dir (str): Directory containing background images.
        objects_dir (str): Directory containing objects.
        output_dir (str): Directory to save output images.
        max_objects (int): Maximum number of objects to blend per background.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all background images
    backgrounds = get_all_files(background_dir)

    # Process each background image
    pbar = tqdm(backgrounds, ncols=150)
    for background_image in pbar:
        pbar.set_description(os.path.basename(background_image))
        process_background(background_image, objects_dir, output_dir, max_objects)

def split_dataset(source, output_dir, train_ratio=0.8):
    """
    Split the dataset into train and validation sets.
    
    Args:
        images_dir (str): Directory containing image files.
        annotations_dir (str): Directory containing YOLO annotation files.
        output_dir (str): Output directory to save train and val sets.
        train_ratio (float): Proportion of data to use for training (default: 0.8).
    """
    # Get all image files
    image_files = get_all_files(source + '/images')
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Create output directories
    train_images_dir = os.path.join(output_dir, "train/images")
    train_annotations_dir = os.path.join(output_dir, "train/labels")
    val_images_dir = os.path.join(output_dir, "val/images")
    val_annotations_dir = os.path.join(output_dir, "val/labels")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_annotations_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_annotations_dir, exist_ok=True)

    def move_files(file_list, dest_images_dir, dest_annotations_dir, mode='train'):
        pbar = tqdm(file_list, ncols=150)
        for image_file in pbar:
            pbar.set_description(f'{mode}')
            annotation_file = os.path.splitext(os.path.basename(image_file))[0] + ".txt"
            annotation_path = os.path.join(source, 'labels', annotation_file)
            if os.path.exists(annotation_path):
                shutil.copy(annotation_path, os.path.join(dest_annotations_dir, annotation_file))
                shutil.copy(image_file, os.path.join(dest_images_dir, os.path.basename(image_file)))

    # Move train and validation files
    move_files(train_files, train_images_dir, train_annotations_dir, mode='train')
    move_files(val_files, val_images_dir, val_annotations_dir, mode='validation')

    print(f"Dataset split completed:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Validation: {len(val_files)} images")


if __name__ == "__main__":
    # target_object = 'hand-luggage'
    # annotation_mode = 'mask'    
    # output_dir = f"/home/appuser/src/output"

    # main_pipeline(
    #     "/home/appuser/src/background",
    #     objects_dir="/home/appuser/src/objects",
    #     output_dir=output_dir,
    # )
    
    # split_dataset(
    #     source=output_dir,
    #     output_dir='/home/appuser/src/data',
    #     train_ratio=0.7,
    # )
    
    
    classes = [
        {
            "class": "wood",
            "class_id": 0,
            "objects": [
                {
                    "name": "woodenplank",
                    "directories": [
                        "/media/appuser/objects/WoodenPlank1",
                        "/media/appuser/objects/WoodenPlank2",
                    ]
                },
                {
                    "name": "pallet",
                    "directories": [
                        "/media/appuser/objects/Pallet1",
                        "/media/appuser/objects/Pallet2",
                        "/media/appuser/objects/Pallet3",
                    ]
                },
                {
                    "name": "woodercraten",
                    "directories": [
                        "/media/appuser/objects/WoodenCrate1",
                    ]
                },
                {
                    "name": "coffeetable",
                    "directories": [
                        "/media/appuser/objects/CoffeeTable1",
                    ]
                }
            ]
        },
        {
            "class": "metal",
            "class_id": 3,
            "objects": [
                {
                    "name": "gasganister",
                    "directories": [
                        "/media/appuser/objects/GasCanister1",
                        "/media/appuser/objects/GasCanister2",
                        "/media/appuser/objects/GasCanister3",
                        "/media/appuser/objects/GasCanister4",
                    ]
                },
                {
                    "name": "fridge",
                    "directories": [
                        "/media/appuser/objects/Fridge1",
                        "/media/appuser/objects/Fridge3",
                        "/media/appuser/objects/Fridge5",
                    ]
                },
                {
                    "name": "ladder",
                    "directories": [
                        "/media/appuser/objects/Ladder1",
                    ]
                },
                {
                    "name": "metalbox",
                    "directories": [
                        "/media/appuser/objects/MetalBox1",
                    ]
                },
                {
                    "name": "metalsheet",
                    "directories": [
                        "/media/appuser/objects/MetalSheet1",
                        "/media/appuser/objects/MetalSheet2",
                    ]
                },
                {
                    "name": "metaltray",
                    "directories": [
                        "/media/appuser/objects/MetalTray",
                    ]
                },
                {
                    "name": "metalpipe",
                    "directories": [
                        "/media/appuser/objects/Pipes1",
                    ]
                },
                {
                    "name": "servicecart",
                    "directories": [
                        "/media/appuser/objects/ServiceCart",
                    ]
                },
                {
                    "name": "steelbox",
                    "directories": [
                        "/media/appuser/objects/SteelBoxTruck",
                    ]
                },
                {
                    "name": "steeltank",
                    "directories": [
                        "/media/appuser/objects/SteelTank",
                    ]
                }
            ]
        },
        {
            "class": "plastic",
            "class_id": 5,
            "objects": [
                {
                    "name": "trashbag",
                    "directories": [
                        "/media/appuser/objects/Trashbag",
                        "/media/appuser/objects/Trashbag2",
                    ]
                },
                {
                    "name": "waterbottle",
                    "directories": [
                        "/media/appuser/objects/WaterBottle",                        
                    ]
                },
                {
                    "name": "bucket",
                    "directories": [
                        "/media/appuser/objects/Bucket1",
                        "/media/appuser/objects/Bucket2",
                        "/media/appuser/objects/PlasticBucket",
                        "/media/appuser/objects/PlasticBucket2",
                    ]
                },
                {
                    "name": "plasticdrum",
                    "directories": [
                        "/media/appuser/objects/PlasticDrum1",
                        "/media/appuser/objects/PlasticDrum2",
                        "/media/appuser/objects/PlasticDrum3",
                        "/media/appuser/objects/PlasticDrum4",
                    ]
                },
                {
                    "name": "plasticjerrycan",
                    "directories": [
                        "/media/appuser/objects/PlasticJerrican1",
                        "/media/appuser/objects/PlasticJerrican3",
                        "/media/appuser/objects/PlasticJerrican4",
                    ]
                },
                {
                    "name": "plasticpallet",
                    "directories": [
                        "/media/appuser/objects/PlasticPallet",
                    ]
                }
            ]
        }
    ]
    
    output_dir = "/media/appuser/data"
    os.makedirs(output_dir, exist_ok=True)
    
    for _class in classes:
        target_objects = _class.get("objects")
        print(f"Analysing {_class.get('class')} ... ...")
        for target_obj in target_objects:
            print(f"Extracting {target_obj.get('name')} ... ...")
            directories = target_obj.get('directories')
            for directory in directories:
                files = get_all_files(
                    os.path.join(
                        directory, "images"
                    )
                )
                
                print(f"{len(files)} Found in {directory}")
                pbar = tqdm(files, ncols=150, desc="Processing")
                for file in pbar:
                    image, bbxes = load_image_and_mask(
                        image_path=file,
                        annotation_dir=os.path.join(directory, "labels_bbox"),
                        annotation_mode="bbox",
                    )
                    
                    data = [
                       [ _class['class_id']] + list(bbx) for bbx in bbxes
                    ]
                    lines = (("%g " * len(line)).rstrip() % tuple(line) + "\n" for line in data)
                    
                    image_location = os.path.join(output_dir, 'images')
                    label_location = os.path.join(output_dir, "labels")
                    os.makedirs(image_location, exist_ok=True)
                    os.makedirs(label_location, exist_ok=True)
                    
                    
                    filename = os.path.basename(file).strip(f".{os.path.basename(file).split('.')[-1]}")
                    cv2.imwrite(os.path.join(image_location, filename + '.jpg'), image)
                    with open(label_location + "/" + filename + '.txt', "w") as file:
                        file.writelines(lines)
                        

    split_dataset(
        source=output_dir,
        output_dir='/home/appuser/src/data',
        train_ratio=0.7,
    )