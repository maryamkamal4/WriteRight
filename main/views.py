import os
import re
import tempfile
import cv2
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import requests
from .utils.preprocessing import preprocess
from .utils.encode_images import encode_images
from .utils.add_text import add_text
from .utils.compare_characters import compare_characters
from .utils.perform_ocr import perform_ocr
from .utils.marking2 import marking
from django.http import JsonResponse


# def add_text(image2, overall_similarity, similarities, boxes2, height2):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1.0
#     font_color = (0, 255, 0)
#     thickness = 2
#     line_type = 2
#     text = f"Overall Similarity: {overall_similarity:.2f}"

#     cv2.putText(image2, text, (10, 110), font, font_scale, font_color, thickness, line_type)

#     for similarity_grade, box2 in zip(similarities, boxes2.splitlines()):
#         if similarity_grade == 0.0:
#             box2 = box2.split(" ")
#             x2_min, y2_min, x2_max, y2_max = int(box2[1]), int(box2[2]), int(box2[3]), int(box2[4])
#             cv2.rectangle(image2, (x2_min, height2 - y2_max), (x2_max, height2 - y2_min), (0, 0, 255), 2)
    
#     cv2.imshow('Resized Image 2', image2)
#     cv2.waitKey(0)
    
#     return image2


# def calculate_similarity(img1, img2):

#     img1 = cv2.resize(img1, (200,200)) # the resizing is important for ssim to work
#     img2 = cv2.resize(img2, (200,200))

#     h, w = img1.shape
#     num_parts = 4  # Split into a 4x4 grid for a total of 16 parts
#     part_height = h // num_parts
#     part_width = w // num_parts
    
#     img1_parts = [img1[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
#                   for i in range(num_parts) for j in range(num_parts)]
#     img2_parts = [img2[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
#                   for i in range(num_parts) for j in range(num_parts)]

#     part_similarities = []

#     for part1, part2 in zip(img1_parts, img2_parts):
#         similarity_value = ssim(part1, part2, win_size=min(part1.shape[0], part1.shape[1], 7), gaussian_weights=True, sigma=1.2, use_sample_covariance=False)
#         similarity_grade = 1 + (9 * (similarity_value - 0.4) / 0.6)
#         similarity_grade = (max(1, min(10, similarity_grade))) + 3
#         part_similarities.append(similarity_grade)

#     overall_similarity = np.mean(part_similarities)
#     print(overall_similarity)

#     return overall_similarity


# def compare_characters(char1, char2, box1, box2, image1, height1, image2, height2, similarities, combined_imgs):
#     if char1 == char2:
#         box1 = box1.split(" ")
#         box2 = box2.split(" ")

#         x1_min, y1_min, x1_max, y1_max = int(box1[1]), int(box1[2]), int(box1[3]), int(box1[4])
#         x2_min, y2_min, x2_max, y2_max = int(box2[1]), int(box2[2]), int(box2[3]), int(box2[4])

#         region_of_interest1 = image1[height1 - y1_max:height1 - y1_min, x1_min:x1_max]
#         region_of_interest2 = image2[height2 - y2_max:height2 - y2_min, x2_min:x2_max]

#         # Display region of interest
#         fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#         ax[0].imshow(region_of_interest1, cmap='gray')
#         ax[0].set_title('Region of Interest 1')
#         ax[0].axis('off')
#         ax[1].imshow(region_of_interest2, cmap='gray')
#         ax[1].set_title('Region of Interest 2')
#         ax[1].axis('off')
#         plt.show()

#         similarity_grade = calculate_similarity(region_of_interest1, region_of_interest2)
#         similarities.append(similarity_grade)

#         # combined_img = find_differences(region_of_interest1, region_of_interest2)
#         # combined_imgs.append(combined_img)
        
#         # # Display region of interest
#         # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#         # ax[0].imshow(region_of_interest1, cmap='gray')
#         # ax[0].set_title('Region of Interest 1')
#         # ax[0].axis('off')
#         # ax[1].imshow(region_of_interest2, cmap='gray')
#         # ax[1].set_title('Region of Interest 2')
#         # ax[1].axis('off')
#         # plt.show()
        
#     else:
#         similarity_grade = 0.0
#         similarities.append(similarity_grade)

#     print("Similarity Grade: {:.2f}".format(similarity_grade))
    
    
# def encode_images(image2):

#     # Convert images to base64
#     _, image2_encoded = cv2.imencode('.png', image2)
    
#     image2_base64 = base64.b64encode(image2_encoded).decode('utf-8')

#     return image2_base64


# def perform_ocr(image, config):
#     # Get image dimensions
#     try:
#         if len(image.shape) == 3:
#             height, width, _ = image.shape
#         elif len(image.shape) == 2:
#             height, width = image.shape
#         else:
#             raise ValueError("Invalid image shape")

#     except ValueError as e:
#         print("Error: Unable to get image shape:", e)

#     # Perform OCR on the image
#     text = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), config=config)
#     text = text.replace(" ", "")

#     # Get bounding boxes of characters
#     boxes = pytesseract.image_to_boxes(image, config=config)

#     return text, boxes, image, height, width


# @csrf_exempt
# def image_comparison_view(request):
    
#     if request.method == 'POST':
        
#         # Receive image2 from the frontend
#         image2_file = request.FILES.get('image2')
#         image1_filename = request.POST.get('image1_filename', '')  # Receive image1 filename
        
#         if image2_file is None or not image1_filename:
#             return JsonResponse({'error': 'Both image2 file and image1 filename are required.'}, status=400)

#         # Construct image1 file path
#         image1_file = os.path.join(r"C:\Users\Lenovo\Desktop\FYP\WriteRight\main\Images\Template writing\\", image1_filename + ".jpeg")
        
#         if not os.path.exists(image1_file):
#             return JsonResponse({'error': 'Image1 file does not exist.'}, status=400)

#         similarities = [] 
#         combined_imgs = []
        
#         my_config = r"--psm 6 --oem 3"

#         # Save the uploaded image temporarily
#         with tempfile.NamedTemporaryFile(delete=False) as temp2:
#             temp2.write(image2_file.read())
#             temp2_path = temp2.name
        

#         # Open and display image1
#         image1 = cv2.imread(image1_file)
#         image1_resized = cv2.resize(image1, (400, 400))
#         cv2.imshow('Image 1', image1_resized)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         # Open and display image2
#         image2 = cv2.imread(temp2_path)
#         image2_resized = cv2.resize(image2, (400, 400))
#         cv2.imshow('Image 2', image2_resized)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         preprocessed_image1 = preprocess(image1_file)
#         preprocessed_image2 = preprocess(temp2_path)

#         # Process the resized images
#         text1, boxes1, ocr_image1, height1, width1 = perform_ocr(preprocessed_image1, my_config)
#         text2, boxes2, ocr_image2, height2, width2 = perform_ocr(preprocessed_image2, my_config)


#         pattern = re.compile(r'[a-zA-Z]')

#         # Ensure both texts have the same length (use the shorter length)
#         text1_filtered = ''.join(re.findall(pattern, text1))
#         text2_filtered = ''.join(re.findall(pattern, text2))

#         # Remove spaces from the filtered text
#         text1_filtered = text1_filtered.replace(' ', '')
#         text2_filtered = text2_filtered.replace(' ', '')

#         min_len = min(len(text1_filtered), len(text2_filtered))
#         text1 = text1_filtered[:min_len]
#         text2 = text2_filtered[:min_len]

#         print("Text1: ", text1)
#         print("Text2: ", text2)

#         for box1, box2, char1, char2 in zip(boxes1.splitlines(), boxes2.splitlines(), text1, text2):
#             print("Character 1:", char1)
#             print("Character 2:", char2)
            
#             compare_characters(char1, char2, box1, box2, ocr_image1, height1, ocr_image2, height2, similarities, combined_imgs)
  
#         overall_similarity = np.mean(similarities)

#         textual_image2 = add_text(ocr_image2, overall_similarity, similarities, boxes2, height2)
        
#         image2_base64 = encode_images(textual_image2)

#         context = {
#             'image2_base64': image2_base64,
#             # 'combined_images_base64': combined_images_base64,
#             'overall_similarity': overall_similarity,
#         }
        
#         print("Similarity Grades: ", similarities)
#         print("Overall Similarity (Grade 1 to 10): {:.2f}".format(overall_similarity))

#         # Clean up temporary files
#         os.unlink(temp2_path)

#         return JsonResponse(context, status=200)
#     else:
#         # Handle other request methods if needed
#         return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def image_comparison_view(request):
    
    if request.method == 'POST':

        try:
            firebase_url = request.POST.get('image1', '')
            print("firebaseurl: ", firebase_url)
            image2_file = request.FILES.get('image2')

            # Error handling for missing parameters
            if not firebase_url or not image2_file:
                return JsonResponse({'error': 'Both Firebase URL and image2 file are required.'}, status=400)

            # Send a GET request to the Firebase URL to retrieve image1
            response = requests.get(firebase_url)
            
            if response.status_code == 200:
                # If the request was successful, decode the image data
                image1_data = response.content

                # Save image2 temporarily
                with tempfile.NamedTemporaryFile(delete=False) as temp2:
                    temp2.write(image2_file.read())
                    temp2_path = temp2.name

                # Save image1 temporarily
                with tempfile.NamedTemporaryFile(delete=False) as temp1:
                    temp1.write(image1_data)
                    temp1_path = temp1.name

                # Open and display image1
                image1 = cv2.imread(temp1_path)
                image1_resized = cv2.resize(image1, (400, 400))
                cv2.imshow('Image 1', image1_resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Open and display image2
                image2 = cv2.imread(temp2_path)
                image2_resized = cv2.resize(image2, (400, 400))
                cv2.imshow('Image 2', image2_resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                similarities = [] 
                combined_imgs = []
                
                my_config = r"--psm 6 --oem 3"
                preprocessed_image1 = preprocess(temp1_path)
                preprocessed_image2 = preprocess(temp2_path)

                # Process the resized images
                text1, boxes1, ocr_image1, height1, width1 = perform_ocr(preprocessed_image1, my_config)
                text2, boxes2, ocr_image2, height2, width2 = perform_ocr(preprocessed_image2, my_config)


                pattern = re.compile(r'[a-zA-Z]')

                # Ensure both texts have the same length (use the shorter length)
                text1_filtered = ''.join(re.findall(pattern, text1))
                text2_filtered = ''.join(re.findall(pattern, text2))

                # Remove spaces from the filtered text
                text1_filtered = text1_filtered.replace(' ', '')
                text2_filtered = text2_filtered.replace(' ', '')

                min_len = min(len(text1_filtered), len(text2_filtered))
                text1 = text1_filtered[:min_len]
                text2 = text2_filtered[:min_len]

                print("Text1: ", text1)
                print("Text2: ", text2)

                for box1, box2, char1, char2 in zip(boxes1.splitlines(), boxes2.splitlines(), text1, text2):
                    print("Character 1:", char1)
                    print("Character 2:", char2)
                    
                    compare_characters(char1, char2, box1, box2, ocr_image1, height1, ocr_image2, height2, similarities, combined_imgs)
        
                overall_similarity = np.mean(similarities)

                if (overall_similarity != 0) and (not np.isnan(overall_similarity)) and (overall_similarity is not None):
                    canvas = marking(temp1_path, temp2_path)

                textual_image2 = add_text(ocr_image2, overall_similarity, similarities, boxes2, height2)
                
                image2_base64, canvas_base4 = encode_images(textual_image2, canvas)

                context = {
                    'overall_similarity': overall_similarity,
                    'canvas': canvas_base4,
                    'student_image': image2_base64,
                }
                
                print("Similarity Grades: ", similarities)
                print("Overall Similarity (Grade 1 to 10): {:.2f}".format(overall_similarity))

                # Clean up temporary files
                os.unlink(temp2_path)
                
                return JsonResponse({'success': True, 'message': 'Image displayed successfully.', 'context': context, 'status': 200})
            else:
                return JsonResponse({'success': False, 'message': 'Failed to retrieve image from Firebase URL.'}, status=400)
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)
    else:
        return JsonResponse({'success': False, 'message': 'Only POST requests are allowed.'}, status=405)

        
    #     # Receive image2 from the frontend
    #     image2_file = request.FILES.get('image2')
    #     image1_filename = request.POST.get('image1_filename', '')  # Receive image1 filename
        
    #     if image2_file is None or not image1_filename:
    #         return JsonResponse({'error': 'Both image2 file and image1 filename are required.'}, status=400)

    #     # Construct image1 file path
    #     image1_file = os.path.join(r"C:\Users\Lenovo\Desktop\FYP\WriteRight\main\Images\Template writing\\", image1_filename + ".jpeg")
        
    #     if not os.path.exists(image1_file):
    #         return JsonResponse({'error': 'Image1 file does not exist.'}, status=400)

    #     similarities = [] 
    #     combined_imgs = []
        
    #     my_config = r"--psm 6 --oem 3"

    #     # Save the uploaded image temporarily
    #     with tempfile.NamedTemporaryFile(delete=False) as temp2:
    #         temp2.write(image2_file.read())
    #         temp2_path = temp2.name
        

    #     # Open and display image1
    #     image1 = cv2.imread(image1_file)
    #     image1_resized = cv2.resize(image1, (400, 400))
    #     cv2.imshow('Image 1', image1_resized)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    #     # Open and display image2
    #     image2 = cv2.imread(temp2_path)
    #     image2_resized = cv2.resize(image2, (400, 400))
    #     cv2.imshow('Image 2', image2_resized)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    #     preprocessed_image1 = preprocess(image1_file)
    #     preprocessed_image2 = preprocess(temp2_path)

    #     # Process the resized images
    #     text1, boxes1, ocr_image1, height1, width1 = perform_ocr(preprocessed_image1, my_config)
    #     text2, boxes2, ocr_image2, height2, width2 = perform_ocr(preprocessed_image2, my_config)


    #     pattern = re.compile(r'[a-zA-Z]')

    #     # Ensure both texts have the same length (use the shorter length)
    #     text1_filtered = ''.join(re.findall(pattern, text1))
    #     text2_filtered = ''.join(re.findall(pattern, text2))

    #     # Remove spaces from the filtered text
    #     text1_filtered = text1_filtered.replace(' ', '')
    #     text2_filtered = text2_filtered.replace(' ', '')

    #     min_len = min(len(text1_filtered), len(text2_filtered))
    #     text1 = text1_filtered[:min_len]
    #     text2 = text2_filtered[:min_len]

    #     print("Text1: ", text1)
    #     print("Text2: ", text2)

    #     for box1, box2, char1, char2 in zip(boxes1.splitlines(), boxes2.splitlines(), text1, text2):
    #         print("Character 1:", char1)
    #         print("Character 2:", char2)
            
    #         compare_characters(char1, char2, box1, box2, ocr_image1, height1, ocr_image2, height2, similarities, combined_imgs)
  
    #     overall_similarity = np.mean(similarities)

    #     textual_image2 = add_text(ocr_image2, overall_similarity, similarities, boxes2, height2)
        
    #     image2_base64 = encode_images(textual_image2)

    #     context = {
    #         'image2_base64': image2_base64,
    #         # 'combined_images_base64': combined_images_base64,
    #         'overall_similarity': overall_similarity,
    #     }
        
    #     print("Similarity Grades: ", similarities)
    #     print("Overall Similarity (Grade 1 to 10): {:.2f}".format(overall_similarity))

    #     # Clean up temporary files
    #     os.unlink(temp2_path)

    #     return JsonResponse(context, status=200)
    # else:
    #     # Handle other request methods if needed
    #     return JsonResponse({'error': 'Method not allowed'}, status=405)
