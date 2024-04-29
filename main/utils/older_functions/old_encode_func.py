# def encode_images(image1_resized, image2_resized, combined_images_resized):

#     # Convert images to base64
#     _, image1_encoded = cv2.imencode('.png', image1_resized)
#     _, image2_encoded = cv2.imencode('.png', image2_resized)
    
#     image1_base64 = base64.b64encode(image1_encoded).decode('utf-8')
#     image2_base64 = base64.b64encode(image2_encoded).decode('utf-8')

#     combined_images_base64 = None
#     if combined_images_resized is not None:
#         _, combined_images_encoded = cv2.imencode('.png', combined_images_resized)
#         combined_images_base64 = base64.b64encode(combined_images_encoded).decode('utf-8')

#     return image1_base64, image2_base64, combined_images_base64