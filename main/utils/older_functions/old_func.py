# def image_comparison_view(request):
    
#     similarities = [] 
#     combined_imgs = []
#     processed_images = [] 
    
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

#         if char1 == char2:
#             box1 = box1.split(" ")
#             box2 = box2.split(" ")

#             x1_min, y1_min, x1_max, y1_max = int(box1[1]), int(box1[2]), int(box1[3]), int(box1[4])
#             x2_min, y2_min, x2_max, y2_max = int(box2[1]), int(box2[2]), int(box2[3]), int(box2[4])

#             region_of_interest1 = image1[height1 - y1_max:height1 - y1_min, x1_min:x1_max]
#             region_of_interest2 = image2[height2 - y2_max:height2 - y2_min, x2_min:x2_max]

#             similarity_grade = calculate_similarity(region_of_interest1, region_of_interest2)
#             similarities.append(similarity_grade)

#             combined_img = find_differences(region_of_interest1, region_of_interest2)
#             combined_imgs.append(combined_img)
#         else:
#             similarity_grade = 0.0
#             similarities.append(similarity_grade)

#         print("Similarity Grade: {:.2f}".format(similarity_grade))

#     overall_similarity = np.mean(similarities)

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

#     # display_images(image1, image2, np.hstack(combined_imgs))
#     combined_images = np.hstack(combined_imgs)
#     image1_resized = cv2.resize(image1, (400, 400))
#     image2_resized = cv2.resize(image2, (400, 400))
#     combined_images_resized = cv2.resize(combined_images, (600, 200))
    
#     # Convert images to base64
#     _, image1_encoded = cv2.imencode('.png', image1_resized)
#     _, image2_encoded = cv2.imencode('.png', image2_resized)
#     _, combined_images_encoded = cv2.imencode('.png', combined_images_resized)

#     image1_base64 = base64.b64encode(image1_encoded).decode('utf-8')
#     image2_base64 = base64.b64encode(image2_encoded).decode('utf-8')
#     combined_images_base64 = base64.b64encode(combined_images_encoded).decode('utf-8')

#     # Append base64 encoded images to global lists
#     processed_images.append(image1_base64)
#     processed_images.append(image2_base64)
#     processed_images.append(combined_images_base64)

#     context = {
#         'image1_base64': image1_base64,
#         'image2_base64': image2_base64,
#         'combined_images_base64': combined_images_base64,
#         'overall_similarity': overall_similarity,
#     }
    
#     print("Similarity Grades: ", similarities)
#     print("Overall Similarity (Grade 1 to 10): {:.2f}".format(overall_similarity))

#     # return JsonResponse(context, status=200)
#     return render(request, r'C:/Users/hamxa/Desktop/FYP/WriteRight/main/templates/image_comparison.html', context)

