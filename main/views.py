import base64
from django.shortcuts import render
from django.http import HttpResponseBadRequest, JsonResponse
import cv2
import pytesseract
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

def resize_and_gray(image, size=(150, 150)):
    resized = cv2.resize(image, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


def find_differences_in_quadrants(quad1, quad2):
    (score, diff) = ssim(quad1, quad2, full=True)

    diff = (diff * 255).astype("uint8")
    _, thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [c for c in contours if 200 < cv2.contourArea(c) < 800]

    marked_quad = cv2.cvtColor(quad1, cv2.COLOR_GRAY2BGR)

    if len(contours):
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(marked_quad, (x, y), (x + w, y + h), (0, 0, 255), 4)

    return marked_quad


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


def perform_ocr(image_path, config):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    text = pytesseract.image_to_string(Image.open(image_path), config=config)
    text = text.replace(" ", "")

    boxes = pytesseract.image_to_boxes(image, config=config)

    return text, boxes, image, height, width


# def display_images(image1, image2, combined_images):
#     image1_resized = cv2.resize(image1, (400, 400))
#     image2_resized = cv2.resize(image2, (400, 400))
#     combined_images_resized = cv2.resize(combined_images, (400, 400))

#     cv2.imshow("Student writing", image2_resized)
#     cv2.imshow("Teacher writing", image1_resized)
#     cv2.imshow('End result', combined_images_resized)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

processed_images = []    
        
def image_comparison_view(request):
    my_config = r"--psm 6 --oem 3"

    path_image1 = r"C:\Users\hamxa\Desktop\FYP\WriteRight\main\Images\Student writing\ahmed2.jpeg"
    path_image2 = r"C:\Users\hamxa\Desktop\FYP\WriteRight\main\Images\Template writing\ahmed3.jpeg"

    text1, boxes1, image1, height1, width1 = perform_ocr(path_image1, my_config)
    text2, boxes2, image2, height2, width2 = perform_ocr(path_image2, my_config)

    # Ensure both texts have the same length (use the shorter length)
    min_len = min(len(text1), len(text2))
    text1 = text1[:min_len]
    text2 = text2[:min_len]

    similarities = []  # Store individual character similarities
    combined_imgs = []

    for box1, box2, char1, char2 in zip(boxes1.splitlines(), boxes2.splitlines(), text1, text2):
        print("Character 1: ", char1)
        print("Character 2: ", char2)

        if char1 == char2:
            box1 = box1.split(" ")
            box2 = box2.split(" ")

            x1_min, y1_min, x1_max, y1_max = int(box1[1]), int(box1[2]), int(box1[3]), int(box1[4])
            x2_min, y2_min, x2_max, y2_max = int(box2[1]), int(box2[2]), int(box2[3]), int(box2[4])

            region_of_interest1 = image1[height1 - y1_max:height1 - y1_min, x1_min:x1_max]
            region_of_interest2 = image2[height2 - y2_max:height2 - y2_min, x2_min:x2_max]

            similarity_grade = calculate_similarity(region_of_interest1, region_of_interest2)
            similarities.append(similarity_grade)

            combined_img = find_differences(region_of_interest1, region_of_interest2)
            combined_imgs.append(combined_img)
        else:
            similarity_grade = 0.0
            similarities.append(similarity_grade)

        print("Similarity Grade: {:.2f}".format(similarity_grade))

    overall_similarity = np.mean(similarities)
    print("Similarity Grades: ", similarities)
    print("Overall Similarity (Grade 1 to 10): {:.2f}".format(overall_similarity))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_color = (0, 255, 0)
    thickness = 2
    line_type = 2
    text = f"Overall Similarity: {overall_similarity:.2f}"

    cv2.putText(image2, text, (10, 110), font, font_scale, font_color, thickness, line_type)

    for similarity_grade, box2 in zip(similarities, boxes2.splitlines()):
        if similarity_grade == 0.0:
            box2 = box2.split(" ")
            x2_min, y2_min, x2_max, y2_max = int(box2[1]), int(box2[2]), int(box2[3]), int(box2[4])
            cv2.rectangle(image2, (x2_min, height2 - y2_max), (x2_max, height2 - y2_min), (0, 0, 255), 2)

    # Check if the images were successfully processed
    if not (image1_resized.any() and image2_resized.any() and combined_images.any()):
        return HttpResponseBadRequest('Error processing images')
    
    # display_images(image1, image2, np.hstack(combined_imgs))
    combined_images = np.hstack(combined_imgs)
    image1_resized = cv2.resize(image1, (400, 400))
    image2_resized = cv2.resize(image2, (400, 400))
    combined_images_resized = cv2.resize(combined_images, (600, 200))
    # Convert images to base64
    _, image1_encoded = cv2.imencode('.png', image1_resized)
    _, image2_encoded = cv2.imencode('.png', image2_resized)
    _, combined_images_encoded = cv2.imencode('.png', combined_images_resized)

    image1_base64 = base64.b64encode(image1_encoded).decode('utf-8')
    image2_base64 = base64.b64encode(image2_encoded).decode('utf-8')
    combined_images_base64 = base64.b64encode(combined_images_encoded).decode('utf-8')

    # Append base64 encoded images to global lists
    processed_images.append(image1_base64)
    processed_images.append(image2_base64)
    processed_images.append(combined_images_base64)

    context = {
        'image1_base64': image1_base64,
        'image2_base64': image2_base64,
        'combined_images_base64': combined_images_base64,
    }

    # return render(request, r'C:/Users/hamxa/Desktop/FYP/WriteRight/main/templates/image_comparison.html', context)
    return JsonResponse(context, status=200)

# def image1_api(request):
#     if processed_images:
#         image1_base64 = processed_images[0]
#         return JsonResponse({'image1_base64': image1_base64})
#     else:
#         return JsonResponse({'error': 'Image not processed'})

# def image2_api(request):
#     if len(processed_images) > 1:
#         image2_base64 = processed_images[1]
#         return JsonResponse({'image2_base64': image2_base64})
#     else:
#         return JsonResponse({'error': 'Image not processed'})

# def combined_images_api(request):
#     if len(processed_images) > 2:
#         combined_images_base64 = processed_images[2]
#         return JsonResponse({'combined_images_base64': combined_images_base64})
#     else:
#         return JsonResponse({'error': 'Image not processed'})