import os
import tempfile
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from skimage.metrics import structural_similarity as ssim
import numpy as np
from .utils.encode_images import encode_images
from .utils.add_text_and_boxes import add_text_and_boxes
from .utils.compare_characters import compare_characters
from .utils.perform_ocr import perform_ocr
from PIL import Image


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
            
        print("_______opening___________")    
        img1 = Image.open(image1_file)
        img1.show()    
        img2 = Image.open(temp2_path)
        img2.show()
        
        # Process the received images
        text1, boxes1, image1, height1, width1 = perform_ocr(image1_file, my_config)
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
        os.unlink(temp2_path)

        return JsonResponse(context, status=200)
    else:
        # Handle other request methods if needed
        return JsonResponse({'error': 'Method not allowed'}, status=405)
