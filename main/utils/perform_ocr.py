import cv2
import pytesseract
from PIL import Image

def perform_ocr(image_path, config):
    image = cv2.imread(image_path)
    
    height, width, _ = image.shape

    text = pytesseract.image_to_string(Image.open(image_path), config=config)
    text = text.replace(" ", "")

    boxes = pytesseract.image_to_boxes(image, config=config)

    return text, boxes, image, height, width