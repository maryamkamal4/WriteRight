import pytesseract
from PIL import Image
import cv2


def perform_ocr(image, config):
    # Get image dimensions
    try:
        if len(image.shape) == 3:
            height, width, _ = image.shape
        elif len(image.shape) == 2:
            height, width = image.shape
        else:
            raise ValueError("Invalid image shape")

    except ValueError as e:
        print("Error: Unable to get image shape:", e)

    # Perform OCR on the image
    text = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), config=config)
    text = text.replace(" ", "")

    # Get bounding boxes of characters
    boxes = pytesseract.image_to_boxes(image, config=config)

    return text, boxes, image, height, width