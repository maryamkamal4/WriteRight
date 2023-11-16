import PIL
import pytesseract
import cv2

my_config = r"--psm 6 --oem 3"
text = pytesseract.image_to_string(PIL.Image.open("./images/ahmed3.jpeg"), config=my_config)
print(text)
img = cv2.imread("./images/ahmed3.jpeg")
height, width, _ = img.shape
boxes = pytesseract.image_to_boxes(img, config=my_config)
for box in boxes.splitlines():
    box = box.split(" ")
    img = cv2.rectangle(img, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 255, 0), 2)
    
cv2.imshow("img", img)
cv2.waitKey(0)