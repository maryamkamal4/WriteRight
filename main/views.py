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
# from .utils.marking import marking
from django.http import JsonResponse
from .utils.older_functions.old_marking import marking


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
            print('firebase request was made....')
            print(response.status_code)
            
            if response.status_code == 200:
                # If the request was successful, decode the image data
                image1_data = response.content
                print('inside if statement')

                # Save image2 temporarily
                with tempfile.NamedTemporaryFile(delete=False) as temp2:
                    temp2.write(image2_file.read())
                    temp2_path = temp2.name

                # Save image1 temporarily
                with tempfile.NamedTemporaryFile(delete=False) as temp1:
                    temp1.write(image1_data)
                    temp1_path = temp1.name

                try:
                    # # Open and display image1
                    # image1 = cv2.imread(temp1_path)
                    # image1_resized = cv2.resize(image1, (400, 400))
                    # cv2.imshow('Image 1', image1_resized)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # # Open and display image2
                    # image2 = cv2.imread(temp2_path)
                    # image2_resized = cv2.resize(image2, (400, 400))
                    # cv2.imshow('Image 2', image2_resized)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # similarities = [] 
                    # combined_imgs = []

                    my_config = r"--psm 6 --oem 3"
                    preprocessed_image1 = preprocess(temp1_path)
                    preprocessed_image2 = preprocess(temp2_path)

                    # Process the resized images
                    text1, boxes1, ocr_image1, height1 = perform_ocr(preprocessed_image1, my_config)
                    text2, boxes2, ocr_image2, height2 = perform_ocr(preprocessed_image2, my_config)

                    pattern = re.compile(r'[a-zA-Z]')

                    # Ensure both texts have the same length (use the shorter length)
                    text1_filtered = ''.join(re.findall(pattern, text1))
                    text2_filtered = ''.join(re.findall(pattern, text2))

                    # # Remove spaces from the filtered text
                    # text1_filtered = text1_filtered.replace(' ', '')
                    # text2_filtered = text2_filtered.replace(' ', '')

                    # min_len = min(len(text1_filtered), len(text2_filtered))
                    # text1 = text1_filtered[:min_len]
                    # text2 = text2_filtered[:min_len]

                    print("Text1: ", text1_filtered)
                    print("Text2: ", text2_filtered)

                    for box1, box2, char1, char2 in zip(boxes1.splitlines(), boxes2.splitlines(), text1_filtered, text2_filtered):
                        print("Character 1:", char1)
                        print("Character 2:", char2)

                        try:
                            similarity_grade = compare_characters(char1, char2, box1, box2, ocr_image1, height1, ocr_image2, height2)
                            canvas = marking(temp1_path, temp2_path)
                            # textual_image2 = add_text(ocr_image2, similarity_grade)
                            canvas_base4 = encode_images(canvas)
                        except Exception as compare_error:
                            print("Error during character comparison:", str(compare_error))
                        break

                    context = {
                        'overall_similarity': similarity_grade,
                        'canvas': canvas_base4,
                    }

                    print("Overall Similarity (Grade 1 to 10): {:.2f}".format(similarity_grade))

                    # Clean up temporary files
                    os.unlink(temp2_path)

                    return JsonResponse({'success': True, 'message': 'Image displayed successfully.', 'context': context, 'status': 200})

                except Exception as img_process_error:
                    return JsonResponse({'success': False, 'message': str(img_process_error)}, status=500)

            else:
                return JsonResponse({'success': False, 'message': 'Failed to retrieve image from Firebase URL.'}, status=400)

        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)

    else:
        return JsonResponse({'success': False, 'message': 'Only POST requests are allowed.'}, status=405)
