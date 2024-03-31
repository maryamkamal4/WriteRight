import os
import tempfile
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest, JsonResponse
from skimage.metrics import structural_similarity as ssim
import numpy as np
from .utils.encode_images import encode_images
from .utils.add_text_and_boxes import add_text_and_boxes
from .utils.compare_characters import compare_characters
from .utils.perform_ocr import perform_ocr


@csrf_exempt
def image_comparison_view(request):
    if request.method == 'POST':
        
        # Receive images from the frontend
        image1_file = request.FILES.get('image1')
        image2_file = request.FILES.get('image2')
        
        if image1_file is None or image2_file is None:
            return JsonResponse({'error': 'Both image1 and image2 files are required.'}, status=400)

        similarities = [] 
        combined_imgs = []
        
        my_config = r"--psm 6 --oem 3"

        # Save the uploaded images temporarily and pass their paths to perform_ocr
        with tempfile.NamedTemporaryFile(delete=False) as temp1, tempfile.NamedTemporaryFile(delete=False) as temp2:
            temp1.write(image1_file.read())
            temp2.write(image2_file.read())
            temp1_path = temp1.name
            temp2_path = temp2.name
        
        # Process the received images
        text1, boxes1, image1, height1, width1 = perform_ocr(temp1_path, my_config)
        text2, boxes2, image2, height2, width2 = perform_ocr(temp2_path, my_config)

        # Ensure both texts have the same length (use the shorter length)
        min_len = min(len(text1), len(text2))
        text1 = text1[:min_len]
        text2 = text2[:min_len]   

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
        os.unlink(temp1_path)
        os.unlink(temp2_path)

        return JsonResponse(context, status=200)
    else:
        # Handle other request methods if needed
        return JsonResponse({'error': 'Method not allowed'}, status=405)
