import base64
import cv2


def encode_images(image2, canvas):

    # Convert images to base64
    _, image2_encoded = cv2.imencode('.png', image2)
    _, canvas_encoded = cv2.imencode('.png', canvas)
    
    image2_base64 = base64.b64encode(image2_encoded).decode('utf-8')
    canvas_base4 = base64.b64encode(canvas_encoded).decode('utf-8')

    return image2_base64, canvas_base4