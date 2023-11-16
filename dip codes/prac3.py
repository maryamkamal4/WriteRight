import cv2
import pytesseract
import numpy as np
from PIL import Image
from scipy.spatial.distance import directed_hausdorff

my_config = r"--psm 6 --oem 3"

# Load the images
image1 = cv2.imread("./images/ahmed2.jpeg")
image2 = cv2.imread("./images/ahmed3.jpeg")
height1, width1, _ = image1.shape
height2, width2, _ = image2.shape
# Perform OCR to get the text from image1
text1 = pytesseract.image_to_string(Image.open("./images/ahmed2.jpeg"), config=my_config)

# Perform OCR to get the text from image2
text2 = pytesseract.image_to_string(Image.open("./images/ahmed3.jpeg"), config=my_config)

# Get the bounding box coordinates of the detected text in image1
detection_boxes1 = pytesseract.image_to_boxes(image1, config=my_config)
x1_min = width1
y1_min = height1
x1_max = 0
y1_max = 0

for box in detection_boxes1.splitlines():
    box = box.split()
    x1_min = min(x1_min, int(box[1]))
    y1_min = min(y1_min, int(box[2]))
    x1_max = max(x1_max, int(box[3]))
    y1_max = max(y1_max, int(box[4]))

# Get the bounding box coordinates of the detected text in image2
detection_boxes2 = pytesseract.image_to_boxes(image2, config=my_config)
x2_min = width2
y2_min = height2
x2_max = 0
y2_max = 0

for box in detection_boxes2.splitlines():
    box = box.split()
    x2_min = min(x2_min, int(box[1]))
    y2_min = min(y2_min, int(box[2]))
    x2_max = max(x2_max, int(box[3]))
    y2_max = max(y2_max, int(box[4]))

# Draw bounding boxes around the detected text in both images
image1_with_box = cv2.rectangle(image1, (x1_min, height1 - y1_max), (x1_max, height1 - y1_min), (0, 255, 0), 2)
image2_with_box = cv2.rectangle(image2, (x2_min, height2 - y2_max), (x2_max, height2 - y2_min), (0, 255, 0), 2)

# Crop the regions within the bounding boxes for further processing
region_of_interest1 = image1[height1 - y1_max:height1 - y1_min, x1_min:x1_max]
region_of_interest2 = image2[height2 - y2_max:height2 - y2_min, x2_min:x2_max]

# Convert both regions to grayscale
gray1 = cv2.cvtColor(region_of_interest1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(region_of_interest2, cv2.COLOR_BGR2GRAY)

# Use thresholding or edge detection to emphasize handwriting strokes within the bounding boxes
_, thresh1 = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY)
_, thresh2 = cv2.threshold(gray2, 128, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded images within the bounding boxes
contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize lists to store directed Hausdorff distances
directed_hausdorff_distances = []

# Calculate the directed Hausdorff distance for each pair of contours within the bounding boxes
for contour1 in contours1:
    for contour2 in contours2:
        directed_hausdorff_distances.append(directed_hausdorff(contour1[:, 0, :], contour2[:, 0, :])[0])

# Calculate an overall similarity score (you may need to adjust this based on your requirements)
# For example, you can use the average directed Hausdorff distance
if directed_hausdorff_distances:
    avg_directed_hausdorff_distance = np.mean(directed_hausdorff_distances)
else:
    avg_directed_hausdorff_distance = float('inf')

# Calculate a similarity score based on the average directed Hausdorff distance
# You may want to adjust the scoring logic based on your requirements
similarity_score = 1 / (1 + avg_directed_hausdorff_distance)

print("Similarity Score:", similarity_score)

# Display the images with bounding boxes
cv2.imshow("Image 1 with Bounding Box", region_of_interest1)
cv2.imshow("Image 2 with Bounding Box", region_of_interest2)
cv2.waitKey(0)
cv2.destroyAllWindows()
