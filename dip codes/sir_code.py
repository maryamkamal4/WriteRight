import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
# TODO add contour detection for enhanced accuracy

def removeWhiteSpace(img):
    # img = cv2.imread('ws.png') # Read in the image and convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = img[y:y+h, x:x+w]
    return rect
    
def match(path1, path2):
    # read the images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = removeWhiteSpace(img1)
    img2 = removeWhiteSpace(img2)
    # turn images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # resize images for comparison
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))
    # display both images
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    similarity_value = "{:.2f}".format(ssim(img1, img2,gaussian_weights = True,sigma= 1.2,use_sample_covariance = False)*100)
    # print("answer is ", float(similarity_value),
    #       "type=", type(similarity_value))
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (150,280)
    fontScale              = 0.60
    fontColor              = (0,0,0)
    thickness              = 1
    lineType               = 2

    cv2.putText(img1,'Score'+similarity_value, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)

    cv2.imshow("Student writing", img1)
    cv2.imshow("Teacher writing", img2)



    cv2.waitKey(0)
    return float(similarity_value)



path1 = './images/ahmed2.jpeg'
path2 = './images/ahmed3.jpeg'
similarity_value = match(path1,path2)
print(similarity_value)
