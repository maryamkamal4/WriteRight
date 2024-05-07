import base64
import cv2


def encode_images(image2):

    # Convert images to base64
    _, image2_encoded = cv2.imencode('.png', image2)
    
    image2_base64 = base64.b64encode(image2_encoded).decode('utf-8')

    return image2_base64