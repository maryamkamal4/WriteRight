# Calculating similarity using ssim + assigning value 0.0 to unequal characters + marking different characters
import cv2
import pytesseract
import PIL.Image
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Function to remove white space around an image
def removeWhiteSpace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y:y+h, x:x+w]
    return rect

# Function to calculate similarity between two images using SSIM
def match(img1, img2):
    img1 = removeWhiteSpace(img1)
    img2 = removeWhiteSpace(img2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (200, 200))
    img2 = cv2.resize(img2, (200, 200))
    
    # Perform SSIM comparison on the whole images (optional)
    similarity_value = ssim(img1, img2, gaussian_weights=True, sigma=1.2, use_sample_covariance=False)
    
    # Map the similarity value to a grade from 1 to 10
    similarity_grade = 1 + (9 * (similarity_value - 0.4) / 0.6)
    similarity_grade = max(1, min(10, similarity_grade))  # Ensure the grade is in the range [1, 10]

    img1 = cv2.resize(img1, (250, 250))
    img2 = cv2.resize(img2, (250, 250))
    
    # Add similarity grade to the images
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5, 200)
    fontScale = 1.0
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2

    cv2.putText(img2, 'Grade: {:.2f}'.format(similarity_grade), 
                bottomLeftCornerOfText, 
                font, 
                fontScale, 
                fontColor, 
                thickness, 
                lineType)

    cv2.imshow("Student writing", img2)
    cv2.imshow("Teacher writing", img1)

    cv2.waitKey(0)
    return similarity_grade

# OCR configuration
my_config = r"--psm 6 --oem 3"

# Load the images
image1 = cv2.imread("./images/ahmed2.jpeg")
image2 = cv2.imread("./images/ahold.jpeg")

height1, width1, _ = image1.shape
height2, width2, _ = image2.shape

# Perform OCR to get text and bounding box coordinates for individual characters in image1
text1 = pytesseract.image_to_string(PIL.Image.open("./images/ahmed2.jpeg"), config=my_config)
print("Text1: ", text1)

# Perform OCR to get text and bounding box coordinates for individual characters in image2
text2 = pytesseract.image_to_string(PIL.Image.open("./images/ahold.jpeg"), config=my_config)
print("Text2: ", text2)

# Get bounding box coordinates for both images
boxes1 = pytesseract.image_to_boxes(image1, config=my_config)
boxes2 = pytesseract.image_to_boxes(image2, config=my_config)

# Ensure both texts have the same length (use the shorter length)
min_len = min(len(text1), len(text2))
text1 = text1[:min_len]
text2 = text2[:min_len]

similarities = []  # Store individual character similarities

# Loop through bounding boxes and calculate similarity only for matching characters
for box1, box2, char1, char2 in zip(boxes1.splitlines(), boxes2.splitlines(), text1, text2):
    
    print("Character 1: ", char1)
    print("Character 2: ", char2)
    # Check if the corresponding characters match
    if char1 == char2:
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
        similarity_grade = match(region_of_interest1, region_of_interest2)
        similarities.append(similarity_grade)
    else:
        similarity_grade = 0.0  # Set the grade to 1 for non-matching characters
        similarities.append(similarity_grade)

    print("Similarity Grade: {:.2f}".format(similarity_grade))

# Calculate the overall similarity as the mean of individual character grades
overall_similarity = np.mean(similarities)
print("Similarity Grades: ", similarities)
print("Overall Similarity (Grade 1 to 10): {:.2f}".format(overall_similarity))

# Display the overall similarity on the original images
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2.5
font_color = (0, 255, 0)
thickness = 2
line_type = 2
text = f"Overall Similarity: {overall_similarity:.2f}"

image3 = image2.copy()
cv2.putText(image2, text, (10, 110), font, font_scale, font_color, thickness, line_type)

# Mark regions with similarity equal to 0 with a rectangle on the last image
for similarity_grade, box2 in zip(similarities, boxes2.splitlines()):
    if similarity_grade == 0.0:  
        box2 = box2.split(" ")
        x2_min, y2_min, x2_max, y2_max = int(box2[1]), int(box2[2]), int(box2[3]), int(box2[4])
        cv2.rectangle(image3, (x2_min, height2 - y2_max), (x2_max, height2 - y2_min), (0, 0, 255), 2)

# Display the images with bounding boxes and overall similarity
image1 = cv2.resize(image1, (400, 400))
image2 = cv2.resize(image2, (400, 400))
image3 = cv2.resize(image3, (400, 400))
cv2.imshow("Student writing", image2)
cv2.imshow("Teacher writing", image1)
cv2.imshow("Marked Differences", image3)

cv2.waitKey(0)
cv2.destroyAllWindows()