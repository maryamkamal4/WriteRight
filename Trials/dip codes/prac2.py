import pytesseract
import cv2
import numpy as np
from PIL import Image

my_config = r"--psm 6 --oem 3"

# Load the image
image = cv2.imread("./images/maryam1.jpeg")
height, width, _ = image.shape

# Perform OCR to get the text
text = pytesseract.image_to_string(Image.open("./images/maryam1.jpeg"), config=my_config)

# Get the bounding box coordinates of the entire text
detection_boxes = pytesseract.image_to_boxes(image, config=my_config)
x_min = width
y_min = height
x_max = 0
y_max = 0

for box in detection_boxes.splitlines():
    box = box.split()
    x_min = min(x_min, int(box[1]))
    y_min = min(y_min, int(box[2]))
    x_max = max(x_max, int(box[3]))
    y_max = max(y_max, int(box[4]))

# Draw a bounding box around the entire detected text
image_with_box = cv2.rectangle(image, (x_min, height - y_max), (x_max, height - y_min), (0, 255, 0), 2)

# Display the image with the bounding box
cv2.imshow("Image with Bounding Box", image_with_box)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the detected text
print("Detected Text:")
print(text)
