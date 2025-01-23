import cv2
import numpy as np

def load_yolo_segmentation_labels(label_file, image_shape):
    """
    Load YOLO segmentation labels and convert them to pixel polygons.
    """
    height, width = image_shape[:2]
    polygons = []

    with open(label_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])  # Currently unused
            points = list(map(float, parts[1:]))

            # Convert normalized coordinates to pixel coordinates
            polygon = [(int(points[i] * width), int(points[i + 1] * height)) for i in range(0, len(points), 2)]
            polygons.append(polygon)

    return polygons

def extract_polygon(mask, threshold_value=10, epsilon_factor=0.01, debug=False):
    """
    Extract the polygon of an object from an image.
    
    Args:
        image_path (str): Path to the input image.
        threshold_value (int): Threshold value for binarization (default: 10).
        epsilon_factor (float): Approximation factor for polygon (default: 0.01).
        debug (bool): If True, display intermediate steps and result.
        
    Returns:
        tuple: A tuple containing:
            - polygon (np.ndarray): Array of polygon points.
            - output_image (np.ndarray): Image with the polygon drawn on it.
    """
    
    h, w = mask.shape
    _, binary_mask = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No object detected in the image.")

    valid_contours = [c for c in contours if cv2.contourArea(c) >= 500 or cv2.contourArea(c) < h * w]
    if not valid_contours:
        raise ValueError("No valid object detected in the image.")

    # Get the largest contour (assuming it's the object)
    largest_contour = max(valid_contours, key=cv2.contourArea)
    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    
    print(polygon)
    output_image = mask.copy()
    
    cv2.polylines(output_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    if debug:
        import matplotlib.pyplot as plt
        
        # Display the binary mask
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Binary Mask")
        plt.axis('off')

        # Display the polygon overlaid on the image
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title("Polygon Extraction")
        plt.axis('off')
        plt.show()

    return polygon.reshape(1, -1, 2), output_image



def get_bounding_box_from_mask(mask, offset):
    """
    Extract bounding box from the mask (binary image) and adjust for offset.
    Args:
        mask: Binary mask of the object.
        offset: (x_offset, y_offset) location where the object is placed in the new image.
    Returns:
        Bounding box (x, y, x, y) adjusted for the offset.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Adjust the bounding box by the offset
        x = offset[0] - w / 2
        y = offset[1] - h / 2
        return (int(x), int(y), int(x + w), int(y + h))
    return None

def draw_polygon_bounding_box_from_mask(image, bounding_box, color=(0, 255, 0), thickness=3):
    """
    Extract the polygon from the mask and draw a bounding box around it.
    Args:
        image: Target image where the bounding box will be drawn.
        mask: Binary mask of the transformed object.
        color: BGR color of the bounding box.
        thickness: Thickness of the bounding box lines.
    """
    if bounding_box:
        # Draw bounding box around the object
        xmin, ymin, xmax, ymax = bounding_box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

def extract_object(image, polygon):
    """
    Extract the object from the image using the polygon.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], (255))
    extracted = cv2.bitwise_and(image, image, mask=mask)
    return extracted, mask


if __name__ == "__main__":
    image_path = '/home/appuser/src/objects/tire_4.5m/masks/panoptic_World-Camera_clgqmbbw.0000.png'

    image = cv2.imread(image_path)
    # Extract the polygon and visualize the result
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    polygon_points, polygon_image = extract_polygon(mask, threshold_value=10 , epsilon_factor=0.01, debug=True)
    
    cv2.imwrite('test.jpg', polygon_image)

    # Print the polygon points
    print("Extracted Polygon Points:", polygon_points)