import base64
import os
import tempfile
import cv2
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
# from .utils.encode_images import encode_images
# from .utils.add_text_and_boxes import add_text_and_boxes
# from .utils.compare_characters import compare_characters
# from .utils.perform_ocr import perform_ocr
from PIL import Image
import pytesseract


def add_text_and_boxes(image1, image2, overall_similarity, similarities, boxes2, height2, combined_imgs):
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
    
    if combined_imgs:
        combined_images = np.hstack(combined_imgs)
        combined_images_resized = cv2.resize(combined_images, (600, 200))
    else:
        combined_images_resized = None
        
    image1_resized = cv2.resize(image1, (400,400))
    image2_resized = cv2.resize(image2, (400, 400))
    cv2.imshow('Resized Image 2', image2_resized)
    cv2.waitKey(0)
    
    return image1_resized, image2_resized, combined_images_resized


def resize_and_gray(image, size=(150, 150)):
    resized = cv2.resize(image, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


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


def compare_characters(char1, char2, box1, box2, image1, height1, image2, height2, similarities, combined_imgs):
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
        
        # Display region of interest
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(region_of_interest1, cmap='gray')
        ax[0].set_title('Region of Interest 1')
        ax[0].axis('off')
        ax[1].imshow(region_of_interest2, cmap='gray')
        ax[1].set_title('Region of Interest 2')
        ax[1].axis('off')
        plt.show()
        
    else:
        similarity_grade = 0.0
        similarities.append(similarity_grade)

    print("Similarity Grade: {:.2f}".format(similarity_grade))
    

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


def display_images(image1, image2, combined_images):
    image1_resized = cv2.resize(image1, (400, 400))
    image2_resized = cv2.resize(image2, (400, 400))
    combined_images_resized = cv2.resize(combined_images, (400, 400))

    cv2.imshow("Student writing", image2_resized)
    cv2.imshow("Teacher writing", image1_resized)
    cv2.imshow('End result', combined_images_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def encode_images(image1_resized, image2_resized, combined_images_resized):

    # Convert images to base64
    _, image1_encoded = cv2.imencode('.png', image1_resized)
    _, image2_encoded = cv2.imencode('.png', image2_resized)
    
    image1_base64 = base64.b64encode(image1_encoded).decode('utf-8')
    image2_base64 = base64.b64encode(image2_encoded).decode('utf-8')

    combined_images_base64 = None
    if combined_images_resized is not None:
        _, combined_images_encoded = cv2.imencode('.png', combined_images_resized)
        combined_images_base64 = base64.b64encode(combined_images_encoded).decode('utf-8')

    return image1_base64, image2_base64, combined_images_base64


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

def perform_ocr(image_path, config):
    image = cv2.imread(image_path)
    
    height, width, _ = image.shape

    text = pytesseract.image_to_string(Image.open(image_path), config=config)
    text = text.replace(" ", "")

    boxes = pytesseract.image_to_boxes(image, config=config)

    return text, boxes, image, height, width

@csrf_exempt
def image_comparison_view(request):
    
    if request.method == 'POST':
        
        # Receive image2 from the frontend
        image2_file = request.FILES.get('image2')
        image1_filename = request.POST.get('image1_filename', '')  # Receive image1 filename
        
        if image2_file is None or not image1_filename:
            return JsonResponse({'error': 'Both image2 file and image1 filename are required.'}, status=400)

        # Construct image1 file path
        image1_file = os.path.join(r"C:\Users\hamxa\Desktop\FYP\WriteRight\main\Images\Template writing\\", image1_filename + ".jpeg")
        
        if not os.path.exists(image1_file):
            return JsonResponse({'error': 'Image1 file does not exist.'}, status=400)

        similarities = [] 
        combined_imgs = []
        
        my_config = r"--psm 6 --oem 3"

        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp2:
            temp2.write(image2_file.read())
            temp2_path = temp2.name
        
        # Process the resized images
        text1, boxes1, image1, height1, width1 = perform_ocr(image1_file, my_config)
        text2, boxes2, image2, height2, width2 = perform_ocr(temp2_path, my_config)


        # Ensure both texts have the same length (use the shorter length)
        min_len = min(len(text1), len(text2))
        text1 = text1[:min_len]
        text2 = text2[:min_len]
        print("Text1: ", text1)
        print("/nText2: ", text2)

        for box1, box2, char1, char2 in zip(boxes1.splitlines(), boxes2.splitlines(), text1, text2):
            print("Character 1: ", char1)
            print("Character 2: ", char2)
            
            compare_characters(char1, char2, box1, box2, image1, height1, image2, height2, similarities, combined_imgs)
  
        overall_similarity = np.mean(similarities)

        image1_resized,image2_resized, combined_images_resized = add_text_and_boxes(image1, image2, overall_similarity, similarities, boxes2, height2, combined_imgs)
        
        image1_base64, image2_base64, combined_images_base64 = encode_images(image1_resized, image2_resized, combined_images_resized)

        context = {
            'image1_base64': image1_base64,
            'image2_base64': image2_base64,
            'combined_images_base64': combined_images_base64,
            'overall_similarity': overall_similarity,
        }
        
        print("Similarity Grades: ", similarities)
        print("Overall Similarity (Grade 1 to 10): {:.2f}".format(overall_similarity))

        # Clean up temporary files
        os.unlink(temp2_path)

        return JsonResponse(context, status=200)
    else:
        # Handle other request methods if needed
        return JsonResponse({'error': 'Method not allowed'}, status=405)
