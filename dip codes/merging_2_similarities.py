import cv2
import pytesseract
import PIL.Image
from skimage.metrics import structural_similarity as ssim
import numpy as np

def match_whole(image1, image2):
    # turn images to grayscale
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # resize images for comparison
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))
    
    # Calculate SSIM
    similarity = ssim(img1, img2, gaussian_weights=True, sigma=1.2, use_sample_covariance=False)

    # Map SSIM to the 1-10 range
    mapped_similarity = 1 + 9 * (similarity + 1) / 2

    return float(mapped_similarity)


# Function to calculate similarity between two images using SSIM
def match_parts(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))
    
    # Split the images into 256 equal parts
    h, w = img1.shape
    num_parts = 16  # Split into a 16x16 grid
    part_height = h // num_parts
    part_width = w // num_parts
    img1_parts = [img1[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width] for i in range(num_parts) for j in range(num_parts)]
    img2_parts = [img2[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width] for i in range(num_parts) for j in range(num_parts)]
    
    part_similarities = []

    for part1, part2 in zip(img1_parts, img2_parts):
        # Perform SSIM comparison on each part
        similarity_value = ssim(part1, part2, gaussian_weights=True, sigma=1.2, use_sample_covariance=False)
        
        # Map the similarity value to a grade from 1 to 10
        similarity_grade = 1 + (9 * (similarity_value - 0.4) / 0.6)
        similarity_grade = max(1, min(10, similarity_grade))  # Ensure the grade is in the range [1, 10]
        
        part_similarities.append(similarity_grade)

    # Calculate the overall similarity for the character as the mean of part similarities
    overall_similarity = np.mean(part_similarities)

    return overall_similarity


# OCR configuration
my_config = r"--psm 6 --oem 3"

# Load the images
image1 = cv2.imread("./images/BUGER1.jpeg")
image2 = cv2.imread("./images/BUgER.jpeg")

similarity_value = match_whole(image1, image2)
print("Whole picture Similarity score: ", similarity_value)

image1 = cv2.resize(image1, (300, 300))
image2 = cv2.resize(image2, (300, 300))

height1, width1, _ = image1.shape
height2, width2, _ = image2.shape

# Perform OCR to get text and bounding box coordinates for individual characters in image1
text1 = pytesseract.image_to_string(PIL.Image.open("./images/BUGER1.jpeg"), config=my_config)

text1 = text1.replace(" ", "")
print("Text1: ", text1)


# Perform OCR to get text and bounding box coordinates for individual characters in image2
text2 = pytesseract.image_to_string(PIL.Image.open("./images/BUgER.jpeg"), config=my_config)

text2 = text2.replace(" ", "")
print("Text2: ", text2)

# Get bounding box coordinates for both images
boxes1 = pytesseract.image_to_boxes(image1, config=my_config)
boxes2 = pytesseract.image_to_boxes(image2, config=my_config)

# Ensure both texts have the same length (use the shorter length)
min_len = min(len(text1), len(text2))
text1 = text1[:min_len]
text2 = text2[:min_len]

similarities = []  # Store individual character similarities

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
        similarity_grade = match_parts(region_of_interest1, region_of_interest2)
        similarities.append(similarity_grade)
        
    else:
        similarity_grade = 0.0  # Set the grade to 0 for non-matching characters
        similarities.append(similarity_grade)

    print("Similarity Grade: {:.2f}".format(similarity_grade))
    

# Calculate the overall similarity as the mean of individual character grades
overall_similarity = np.mean(similarities)
print("Similarity score of image parts using ocr: ", overall_similarity)

all_similarities = [overall_similarity, similarity_value]
final_similarity = np.mean(all_similarities) 
print("Final Similarity (Grade 1 to 10): {:.2f}".format(final_similarity))

# Define the weights for overall similarity and similarity value
weight_overall_similarity = 0.6  # You can adjust this weight as needed
weight_similarity_value = 0.4

# Calculate the final similarity as a weighted average
weighted_similarity = (weight_overall_similarity * overall_similarity) + (weight_similarity_value * similarity_value)
print("Weighted Similarity: ", weighted_similarity)

# Display the overall similarity on the original images
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (0, 255, 0)
thickness = 1
line_type = 2
text = f"Overall Similarity: {final_similarity:.2f}"

cv2.putText(image2, text, (5, 80), font, font_scale, font_color, thickness, line_type)

# Mark regions with similarity equal to 0 with a rectangle on the last image
for similarity_grade, box2 in zip(similarities, boxes2.splitlines()):
    if similarity_grade == 0.0:  
        box2 = box2.split(" ")
        x2_min, y2_min, x2_max, y2_max = int(box2[1]), int(box2[2]), int(box2[3]), int(box2[4])
        cv2.rectangle(image2, (x2_min, height2 - y2_max), (x2_max, height2 - y2_min), (0, 0, 255), 2)

# Display the images with bounding boxes and overall similarity
image1 = cv2.resize(image1, (300, 300))
image2 = cv2.resize(image2, (300, 300))

cv2.imshow("Student writing", image2)
cv2.imshow("Teacher writing", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()