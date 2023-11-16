import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL.Image

import pytesseract

# Path to the Tesseract executable (change this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
my_config = r"--psm 6 --oem 3"


image_file = "./images/burger11.jpeg"
img = cv2.imread(image_file)

def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

# display(image_file)

# ___________________________________ DESKEWING IMAGE___________________________________________
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle

    return angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image for horizontal alignment
def deskewHorizontal(cvImage):
    angle = getSkewAngle(cvImage)
    # Check if the skew angle is significant before correction
    if abs(angle) > 5.0:  # Adjust the threshold as needed
        return rotateImage(cvImage, -0.14 * angle)
    else:
        return cvImage

# Load your image

# display("./images/Hafsa3.jpeg")
# Correct horizontal skew only if it is significant
fixed = deskewHorizontal(img)

# Save the corrected image
cv2.imwrite("./images/horizontal_fixed.jpeg", fixed)
display("./images/horizontal_fixed.jpeg")


fixed_img=cv2.imread("./images/horizontal_fixed.jpeg")

# ____________________________________________________________________________

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def printText(path,imageName):
    text = pytesseract.image_to_string(PIL.Image.open(path), config=my_config)
    print(imageName+text)

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)
def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

inverted_image=cv2.bitwise_not(fixed_img)
path="./images/inverted.jpeg"

cv2.imwrite(path,inverted_image)
# display(path)
inv_image=cv2.imread("./images/horizontal_fixed.jpeg")
inv_image=grayscale(inv_image)
pathGray="./images/invertedGray.jpeg"

thresh, im_bw = cv2.threshold(inv_image, 100, 100, cv2.THRESH_BINARY)
cv2.imwrite(pathGray, im_bw)

no_noise = noise_removal(im_bw)
cv2.imwrite("./images/no_noise.jpeg", no_noise)
# display("./images/no_noise.jpeg")

eroded_image = thin_font(no_noise)
cv2.imwrite("./images/eroded_image.jpeg", eroded_image)

# display("./images/eroded_image.jpeg")

# display(pathGray)

printText(image_file,"WITHOUT INVERSION: ")
# printText(path,"INVERSION: ")
# printText(pathGray,"binarize: ")

printText("./images/no_noise.jpeg","Noisy: ")

printText("./images/eroded_image.jpeg","DILATION: ")




# ___________________________________ REMOVING BORDERS___________________________________________

# display("./images/no_noise.jpeg")
def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

no_borders = remove_borders(no_noise)
cv2.imwrite("./images/no_borders.jpeg", no_borders)
# display('./images/no_borders.jpeg')

# ____________________________________ADDING BORDERS___________________________________________

color = [255, 255, 255]
top, bottom, left, right = [150]*4
image_with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
cv2.imwrite("./images/borders.jpeg", image_with_border)
display("./images/borders.jpeg")


