# Calculating similarity using cosine similarity
import cv2
import pytesseract
import PIL.Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to remove white space around an image
def removeWhiteSpace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y:y+h, x:x+w]
    return rect

# Function to calculate similarity between two images using cosine similarity
def match(img1, img2):
    img1 = removeWhiteSpace(img1)
    img2 = removeWhiteSpace(img2)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Resize the images to a fixed size (e.g., 200x200)
    img1_gray = cv2.resize(img1_gray, (300, 300))
    img2_gray = cv2.resize(img2_gray, (300, 300))
    
    # Flatten the grayscale images to create feature vectors
    feature_vector1 = img1_gray.flatten().reshape(1, -1)
    feature_vector2 = img2_gray.flatten().reshape(1, -1)
    
    # Calculate cosine similarity between the feature vectors
    similarity_value = cosine_similarity(feature_vector1, feature_vector2)[0][0] * 100

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (0, 0, 0)
    thickness = 2
    lineType = 2

    # Position to display the similarity value on the images
    text_position = (10, 30)

    cv2.putText(img1_gray, 'Score: {:.2f}'.format(similarity_value), 
                text_position, 
                font, 
                fontScale, 
                fontColor, 
                thickness, 
                lineType)

    cv2.imshow("Student writing", img1_gray)
    cv2.imshow("Teacher writing", img2_gray)

    cv2.waitKey(0)
    return float(similarity_value)

# OCR configuration
my_config = r"--psm 6 --oem 3"

# Load the images
image1 = cv2.imread("./images/ahmed3.jpeg")
image2 = cv2.imread("./images/maryam2.jpeg")

height1, width1, _ = image1.shape
height2, width2, _ = image2.shape

# Perform OCR to get text and bounding box coordinates for individual characters in image1
text1 = pytesseract.image_to_string(PIL.Image.open("./images/ahmed3.jpeg"), config=my_config)
print("Text1: ", text1)

# Perform OCR to get text and bounding box coordinates for individual characters in image2
text2 = pytesseract.image_to_string(PIL.Image.open("./images/maryam2.jpeg"), config=my_config)
print("Text2: ", text2)

# Get bounding box coordinates for both images
boxes1 = pytesseract.image_to_boxes(image1, config=my_config)
boxes2 = pytesseract.image_to_boxes(image2, config=my_config)

similarities = []  # Store individual character similarities

# Loop through bounding boxes and calculate similarity for each character pair
for box1, box2 in zip(boxes1.splitlines(), boxes2.splitlines()):
    # Split bounding box coordinates
    box1 = box1.split(" ")
    box2 = box2.split(" ")

    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = int(box1[1]), int(box1[2]), int(box1[3]), int(box1[4])
    x2_min, y2_min, x2_max, y2_max = int(box2[1]), int(box2[2]), int(box2[3]), int(box2[4])

    # Crop the regions within the bounding boxes for further processing
    region_of_interest1 = image1[height1 - y1_max:height1 - y1_min, x1_min:x1_max]
    region_of_interest2 = image2[height2 - y2_max:height2 - y2_min, x2_min:x2_max]

    # Calculate similarity for each pair of cropped character regions
    similarity_value = match(region_of_interest1, region_of_interest2)
    similarities.append(similarity_value)

# Calculate the overall similarity as the mean of individual character similarities
overall_similarity = np.mean(similarities)
print("Similarity Array: ", similarities)
print("Overall Similarity:", overall_similarity)

# Display the overall similarity on the original images
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
font_color = (0, 0, 0)
thickness = 2
line_type = 2
text = f"Overall Similarity: {overall_similarity:.2f}"

cv2.putText(image1, text, (10, 30), font, font_scale, font_color, thickness, line_type)
cv2.putText(image2, text, (10, 30), font, font_scale, font_color, thickness, line_type)

# Display the images with bounding boxes and overall similarity
cv2.imshow("Student writing", image1)
cv2.imshow("Teacher writing", image2)

cv2.waitKey(0)
cv2.destroyAllWindows()
