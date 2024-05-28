import os
import tempfile
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import requests
from .utils.preprocessing import preprocess
from .utils.encode_images import encode_images
from .utils.compare_characters import compare_characters
from .utils.perform_ocr import perform_ocr
# from .utils.marking import marking
from django.http import JsonResponse
from .utils.stylus_marking import marking


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

                    my_config = r"--psm 6 --oem 3"
                    
                    preprocessed_image1 = preprocess(temp1_path)
                    preprocessed_image2 = preprocess(temp2_path)

                    # Perform OCR on preprocessed images
                    text1, boxes1, ocr_image1, height1 = perform_ocr(preprocessed_image1, my_config)
                    text2, boxes2, ocr_image2, height2 = perform_ocr(preprocessed_image2, my_config)

                    print("Text1: ", text1)
                    print("Text2: ", text2)

                    # Find the index of the character in text2 that matches char1 from text1
                    char2_index = text2.find(text1)

                    if char2_index != -1:
                        # If a matching character is found in text2, reassign char2 and box2
                        char2 = text2[char2_index]
                        box2 = boxes2.splitlines()[char2_index]
                    else:
                        # If no matching character is found, take the first character and its box from text2
                        char2 = text2[0]
                        box2 = boxes2.splitlines()[0]

                    # Select the corresponding character from text1 and its box
                    char1 = text1[0]  # Assume we're taking the first character for comparison
                    box1 = boxes1.splitlines()[0]  # Take the first box for comparison

                    print("Character 1:", char1)
                    print("Character 2:", char2)

                    try:
                        similarity_grade = compare_characters(char1, char2, box1, box2, ocr_image1, height1, ocr_image2, height2)
                        canvas = marking(temp1_path, temp2_path)
                        canvas_base4 = encode_images(canvas)
                    except Exception as compare_error:
                        print("Error during character comparison:", str(compare_error))

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
