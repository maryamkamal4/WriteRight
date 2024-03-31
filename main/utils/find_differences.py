import cv2
import numpy as np
from .resize_and_gray import resize_and_gray
from .difference_in_quadrants import find_differences_in_quadrants


def find_differences(img1, img2):
    img1 = resize_and_gray(img1)
    img2 = resize_and_gray(img2)

    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    # Split the images into 4 quadrants
    h, w = img1.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    quadrants_img1 = [img1[:mid_h, :mid_w], img1[:mid_h, mid_w:], img1[mid_h:, :mid_w], img1[mid_h:, mid_w:]]
    quadrants_img2 = [img2[:mid_h, :mid_w], img2[:mid_h, mid_w:], img2[mid_h:, :mid_w], img2[mid_h:, mid_w:]]

    marked_quadrants = []

    for quad1, quad2 in zip(quadrants_img1, quadrants_img2):
        marked_quad = find_differences_in_quadrants(quad1, quad2)
        marked_quadrants.append(marked_quad)

    # Combine the marked quadrants into a single image
    combined_image = np.vstack([np.hstack(marked_quadrants[:2]), np.hstack(marked_quadrants[2:])])

    return combined_image
