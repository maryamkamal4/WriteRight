# def image_comparison_view(request):
    
#     similarities = [] 
#     combined_imgs = []
    
#     my_config = r"--psm 6 --oem 3"

#     path_image1 = r"C:\Users\hamxa\Desktop\FYP\WriteRight\main\Images\Student writing\ahmed2.jpeg"
#     path_image2 = r"C:\Users\hamxa\Desktop\FYP\WriteRight\main\Images\Template writing\ahmed3.jpeg"

#     text1, boxes1, image1, height1, width1 = perform_ocr(path_image1, my_config)
#     text2, boxes2, image2, height2, width2 = perform_ocr(path_image2, my_config)

#     # Ensure both texts have the same length (use the shorter length)
#     min_len = min(len(text1), len(text2))
#     text1 = text1[:min_len]
#     text2 = text2[:min_len]   

#     for box1, box2, char1, char2 in zip(boxes1.splitlines(), boxes2.splitlines(), text1, text2):
#         print("Character 1: ", char1)
#         print("Character 2: ", char2)
        
#         compare_characters(char1, char2, box1, box2, image1, height1, image2, height2, similarities, combined_imgs)

#     overall_similarity = np.mean(similarities)

#     image1_resized,image2_resized, combined_images_resized = add_text_and_boxes(image1, image2, overall_similarity, similarities, boxes2, height2, combined_imgs)
    
#     image1_base64, image2_base64, combined_images_base64 = encode_images(image1_resized, image2_resized, combined_images_resized)


#     context = {
#         'image1_base64': image1_base64,
#         'image2_base64': image2_base64,
#         'combined_images_base64': combined_images_base64,
#         'overall_similarity': overall_similarity,
#     }
    
#     print("Similarity Grades: ", similarities)
#     print("Overall Similarity (Grade 1 to 10): {:.2f}".format(overall_similarity))

#     return JsonResponse(context, status=200)

    # return render(request, r'C:/Users/hamxa/Desktop/FYP/WriteRight/main/templates/image_comparison.html', context)