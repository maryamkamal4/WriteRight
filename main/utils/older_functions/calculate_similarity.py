import numpy as np
from skimage.metrics import structural_similarity as ssim
from .resize_and_gray import resize_and_gray

def calculate_similarity(img1, img2):
    img1 = resize_and_gray(img1, size=(200, 200))
    img2 = resize_and_gray(img2, size=(200, 200))

    h, w = img1.shape
    num_parts = 16  # Split into a 16x16 grid
    part_height = h // num_parts
    part_width = w // num_parts
    img1_parts = [img1[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
                  for i in range(num_parts) for j in range(num_parts)]
    img2_parts = [img2[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
                  for i in range(num_parts) for j in range(num_parts)]

    part_similarities = []

    for part1, part2 in zip(img1_parts, img2_parts):
        similarity_value = ssim(part1, part2, gaussian_weights=True, sigma=1.2, use_sample_covariance=False)
        similarity_grade = 1 + (9 * (similarity_value - 0.4) / 0.6)
        similarity_grade = max(1, min(10, similarity_grade))
        part_similarities.append(similarity_grade)

    overall_similarity = np.mean(part_similarities)

    return overall_similarity