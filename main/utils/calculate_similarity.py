import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_similarity(img1, img2):
    try:
        img1 = cv2.resize(img1, (800,800)) # the resizing is important for ssim to work
        img2 = cv2.resize(img2, (800,800))

        # cv2.imshow('Teacher Image', img1)
        # cv2.imshow('Student Image', img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        h, w = img1.shape
        num_parts = 4  # Split into a 4x4 grid for a total of 16 parts
        part_height = h // num_parts
        part_width = w // num_parts
        
        img1_parts = [img1[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
                      for i in range(num_parts) for j in range(num_parts)]
        img2_parts = [img2[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
                      for i in range(num_parts) for j in range(num_parts)]

        part_similarities = []

        for part1, part2 in zip(img1_parts, img2_parts):
            similarity_value = ssim(part1, part2, win_size=min(part1.shape[0], part1.shape[1], 7), gaussian_weights=True, sigma=1.2, use_sample_covariance=False)
            similarity_grade = 1 + (9 * (similarity_value - 0.4) / 0.6)
            similarity_grade = (max(1, min(10, similarity_grade))) + 3
            part_similarities.append(similarity_grade)

        overall_similarity = np.mean(part_similarities)

        return overall_similarity
    
    except Exception as e:
        print("Error during similarity calculation:", str(e))
        return 0.0
