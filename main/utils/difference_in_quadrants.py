import cv2
from skimage.metrics import structural_similarity as ssim

def find_differences_in_quadrants(quad1, quad2):
    (score, diff) = ssim(quad1, quad2, full=True)

    diff = (diff * 255).astype("uint8")
    _, thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [c for c in contours if 200 < cv2.contourArea(c) < 800]

    marked_quad = cv2.cvtColor(quad1, cv2.COLOR_GRAY2BGR)

    if len(contours):
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(marked_quad, (x, y), (x + w, y + h), (0, 0, 255), 4)

    return marked_quad