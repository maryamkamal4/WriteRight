#this code shows comparison images with percentage similarities
import PIL
import cv2
import pytesseract
import numpy as np
from PIL import Image
from skimage.measure import label
from scipy.spatial.distance import cdist
from scipy import ndimage 

# Your existing code for OCR and bounding box detection
my_config = r"--psm 6 --oem 3"

# Load the images
image1 = cv2.imread("./images/ahmed3.jpeg")
image2 = cv2.imread("./images/maryam2.jpeg")
height1, width1, _ = image1.shape
height2, width2, _ = image2.shape

# Perform OCR to get the text and bounding box coordinates for individual characters in image1
text1 = pytesseract.image_to_string(PIL.Image.open("./images/ahmed3.jpeg"), config=my_config)
print("Text1: ", text1)

# Perform OCR to get the text and bounding box coordinates for individual characters in image2
text2 = pytesseract.image_to_string(PIL.Image.open("./images/maryam2.jpeg"), config=my_config)
print("Text2: ", text2)

boxes1 = pytesseract.image_to_boxes(image1, config=my_config)
boxes2 = pytesseract.image_to_boxes(image2, config=my_config)

# Define a function to resize or pad strokes to a common dimension
def resize_or_pad_strokes(strokes, target_shape):
    resized_strokes = []
    for stroke in strokes:
        # Resize or pad the stroke to the target shape
        resized_stroke = cv2.resize(stroke, target_shape, interpolation=cv2.INTER_LINEAR)
        resized_strokes.append(resized_stroke)
    return resized_strokes

# Define a function to calculate DTW distance between two sequences
def dtw_distance(seq1, seq2):
    return cdist([seq1.flatten()], [seq2.flatten()], 'euclidean')[0, 0]

# Define a function to map similarity scores to percentages
def similarity_to_percentage(similarity_score):
    # Assuming that lower scores represent less similarity
    # and higher scores represent more similarity
    # You can adjust this logic based on your specific requirements
    max_score = np.max(similarity_scores)
    min_score = np.min(similarity_scores)
    
    # Check if max_score and min_score are equal to prevent division by zero
    if max_score == min_score:
        return 0.0  # Set a default value (0%) when scores are equal
    
    # Map the similarity score to a percentage scale
    percentage = 100 * (1 - (similarity_score - min_score) / (max_score - min_score))
    
    return percentage

similarity_percentages = []
similarity_scores = []
max_possible_similarities = []

# Define the common shape for resizing or padding strokes
common_shape = (50, 50)

# Create a canvas to hold comparison images vertically
canvas_height = common_shape[0] * 2 * len(boxes1.splitlines())
canvas_width = common_shape[1] * len(boxes1.splitlines())
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Initialize the vertical position for pasting comparison images
y_position = 0

# Iterate through the bounding boxes in both images
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

    # Convert the regions to grayscale
    gray1 = cv2.cvtColor(region_of_interest1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(region_of_interest2, cv2.COLOR_BGR2GRAY)

    # Threshold the regions to get binary images
    _, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Label connected components in the binary images
    components1 = label(thresh1)
    components2 = label(thresh2)

    # Find the bounding boxes of the connected components
    regions1 = ndimage.find_objects(components1)
    regions2 = ndimage.find_objects(components2)

    # Extract the individual handwriting strokes from ROIs
    strokes1 = [gray1[region] for region in regions1]
    strokes2 = [gray2[region] for region in regions2]

    # Resize or pad strokes to a common dimension
    resized_strokes1 = resize_or_pad_strokes(strokes1, common_shape)
    resized_strokes2 = resize_or_pad_strokes(strokes2, common_shape)

    # Resize the regions of interest to match common_shape
    region_of_interest1_resized = cv2.resize(region_of_interest1, common_shape, interpolation=cv2.INTER_LINEAR)
    region_of_interest2_resized = cv2.resize(region_of_interest2, common_shape, interpolation=cv2.INTER_LINEAR)

    # Calculate the DTW distance between all pairs of ROIs
    distances = np.zeros((len(region_of_interest1_resized), len(region_of_interest2_resized)))
    for i, stroke1 in enumerate(region_of_interest1_resized):
        for j, stroke2 in enumerate(region_of_interest2_resized):
            distances[i, j] = dtw_distance(stroke1, stroke2)

    similarity_score = np.min(distances)
    similarity_scores.append(similarity_score)

    # Calculate the maximum possible similarity score (worst-case scenario)
    max_possible_similarity = np.max(distances)
    max_possible_similarities.append(max_possible_similarity)
    
    # Calculate the similarity percentage
    similarity_percentage = 100 * (1 - (similarity_score / max_possible_similarity))

    # Add the similarity percentage to the list
    similarity_percentages.append(similarity_percentage)

    # Initialize an empty image to display ROIs and their comparisons
    comparison_image = np.zeros((common_shape[0] * 2, common_shape[1] * len(boxes1.splitlines()), 3), dtype=np.uint8)

    # Visualize the ROIs and their comparisons
    comparison_image[:common_shape[0], :common_shape[1], :] = region_of_interest1_resized
    comparison_image[common_shape[0]:, :common_shape[1], :] = region_of_interest2_resized
    
    # Convert similarity score to percentage
    similarity_percentage = similarity_to_percentage(similarity_score)
    
    cv2.putText(comparison_image, f"Similarity: {similarity_percentage:.2f}%", (10, common_shape[0] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
    # Paste the comparison image on the canvas
    canvas[y_position:y_position + common_shape[0] * 2, :, :] = comparison_image
    y_position += common_shape[0] * 2


# print("Similarity scores: ", similarity_scores)
# print("Max Similarity scores: ", max_possible_similarities)

# # Calculate the collective similarity percentage as the average of all individual percentages
# collective_similarity_percentage = np.mean([100 * (1 - (score / max_score)) for score, max_score in zip(similarity_scores, max_possible_similarities)])

# # Print the collective similarity percentage
# print("Collective Similarity Percentage:", collective_similarity_percentage)

# Display the canvas with all comparison images
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)