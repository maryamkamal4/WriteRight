import cv2
import pytesseract
import PIL.Image
from skimage.metrics import structural_similarity as ssim
import numpy as np

def find_differences(img1, img2):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Resize ROIs to 200x200
    img1 = cv2.resize(img1_gray, (150, 150))
    img2 = cv2.resize(img2_gray, (150, 150))

    # Compute the structural similarity index for the entire images
    (score, diff) = ssim(img1, img2, full=True)

    # Create a difference image
    diff = (diff * 255).astype("uint8")
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY_INV)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [c for c in contours if 200 < cv2.contourArea(c) < 600]

    # Mark differences in the original images
    marked_img1 = img1.copy()

    if len(contours):
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(marked_img1, (x, y), (x + w, y + h), (0, 0, 255), 4)

    return marked_img1, img2

# OCR configuration
my_config = r"--psm 6 --oem 3"

# Load the images
image1 = cv2.imread("./images/ahoad.jpeg")
image2 = cv2.imread("./images/ahold.jpeg")

height1, width1, _ = image1.shape
height2, width2, _ = image2.shape

# Perform OCR to get text and bounding box coordinates for individual characters in image1
text1 = pytesseract.image_to_string(PIL.Image.open("./images/ahoad.jpeg"), config=my_config)

text1 = text1.replace(" ", "")
print("Text1: ", text1)

# Perform OCR to get text and bounding box coordinates for individual characters in image2
text2 = pytesseract.image_to_string(PIL.Image.open("./images/ahold.jpeg"), config=my_config)

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
combined_student_imgs = []
combined_teacher_imgs = []

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

        # Find differences in the character regions and draw rectangles
        student_imgs, teacher_imgs = find_differences(region_of_interest1, region_of_interest2)
        combined_student_imgs.append(student_imgs)
        combined_teacher_imgs.append(teacher_imgs)

stacked_student_imgs = np.hstack(combined_student_imgs)
stacked_teacher_imgs = np.hstack(combined_teacher_imgs)

cv2.imshow('Student Marked Image', stacked_student_imgs)
cv2.imshow('Teacher Original Image', stacked_teacher_imgs)
cv2.waitKey()
cv2.destroyAllWindows()
