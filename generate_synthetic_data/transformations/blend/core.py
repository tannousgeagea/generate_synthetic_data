import cv2

def execute(background, object_img, mask, center):
    """
    Insert the object using Poisson blending for seamless integration.
    """
    # center = (location[0] + object_img.shape[1] // 2, location[1] + object_img.shape[0] // 2)
    blended_image = cv2.seamlessClone(object_img, background, mask, center, cv2.NORMAL_CLONE)
    return blended_image