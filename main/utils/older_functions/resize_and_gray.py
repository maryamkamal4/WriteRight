import cv2

def resize_and_gray(image, size=(150, 150)):
    resized = cv2.resize(image, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray