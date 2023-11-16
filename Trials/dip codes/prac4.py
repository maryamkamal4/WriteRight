import cv2
import pytesseract
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.ndimage import label, find_objects

my_config = r"--psm 6 --oem 3"

# Load the images
image1 = cv2.imread("./images/ahmed3.jpeg")
image2 = cv2.imread("./images/ahmed2.jpeg")
height1, width1, _ = image1.shape
height2, width2, _ = image2.shape
# Perform OCR to get the text from image1
text1 = pytesseract.image_to_string(Image.open("./images/ahmed3.jpeg"), config=my_config)

# Perform OCR to get the text from image2
text2 = pytesseract.image_to_string(Image.open("./images/ahmed2.jpeg"), config=my_config)

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

# Convert the regions to grayscale
gray1 = cv2.cvtColor(region_of_interest1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(region_of_interest2, cv2.COLOR_BGR2GRAY)

# Threshold the regions to get binary images
_, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Label connected components in the binary images
components1, _ = label(thresh1)
components2, _ = label(thresh2)

# Find the bounding boxes of the connected components
regions1 = find_objects(components1)
regions2 = find_objects(components2)

# Extract the individual handwriting strokes from ROIs
strokes1 = [gray1[region] for region in regions1]
strokes2 = [gray2[region] for region in regions2]

# Define a function to resize or pad strokes to a common dimension
def resize_or_pad_strokes(strokes, target_shape):
    resized_strokes = []
    for stroke in strokes:
        # Resize or pad the stroke to the target shape
        resized_stroke = cv2.resize(stroke, target_shape, interpolation=cv2.INTER_LINEAR)
        resized_strokes.append(resized_stroke)
    return resized_strokes

# Resize or pad strokes to a common dimension
common_shape = (50, 50)  # You can adjust the target shape as needed
resized_strokes1 = resize_or_pad_strokes(strokes1, common_shape)
resized_strokes2 = resize_or_pad_strokes(strokes2, common_shape)

print("resized_strokes1", resized_strokes1)
print("resized_strokes2", resized_strokes2)

# Define a function to calculate DTW distance between two sequences
def dtw_distance(seq1, seq2):
    return cdist([seq1.flatten()], [seq2.flatten()], 'euclidean')[0, 0]

# Calculate the DTW distance between all pairs of resized strokes
distances = np.zeros((len(resized_strokes1), len(resized_strokes2)))
for i, stroke1 in enumerate(resized_strokes1):
    for j, stroke2 in enumerate(resized_strokes2):
        distances[i, j] = dtw_distance(stroke1, stroke2)

print("Distances: ", distances)

# Calculate a similarity score by finding the minimum distance
similarity_score = np.min(distances)

# Print the similarity score
print("Similarity Score:", similarity_score)

# Calculate the maximum possible similarity score (worst-case scenario)
max_possible_similarity = np.max(distances)
print("Max score", max_possible_similarity)

# Calculate the similarity percentage
similarity_percentage = 100 * (1 - (similarity_score / max_possible_similarity))

# Print the similarity percentage
print("Similarity Percentage:", similarity_percentage)

# Display the images with bounding boxes
cv2.imshow("Image 1 with Bounding Box", region_of_interest1)
cv2.imshow("Image 2 with Bounding Box", region_of_interest2)
cv2.waitKey(0)
cv2.destroyAllWindows()