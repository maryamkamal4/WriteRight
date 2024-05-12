import cv2
import numpy as np
from preprocessing import preprocess

def marking(image1_path, image2_path):
    # Read images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # image1 = preprocess(image1_path)
    # image2 = preprocess(image2_path)

    # Perform edge detection
    edges1 = cv2.Canny(gray1, 100, 200)
    edges2 = cv2.Canny(gray2, 100, 200)

    # Find contours
    contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour from each image
    max_contour1 = max(contours1, key=cv2.contourArea)
    max_contour2 = max(contours2, key=cv2.contourArea)

    def resize_contour_with_aspect_ratio(contour, target_width, target_height):
        # Ensure contour has two dimensions
        if contour.ndim == 3:
            contour = contour.squeeze(1)

        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate scaling factors to fit the contour within the target dimensions
        scale_factor_w = target_width / w
        scale_factor_h = target_height / h
        scale_factor = min(scale_factor_w, scale_factor_h)

        # Scale the contour while maintaining aspect ratio
        resized_contour = contour * scale_factor

        # Translate the contour to match the center of the bounding rectangle to the center of the canvas
        centroid_x = (x + w / 2)
        centroid_y = (y + h / 2)
        resized_contour[:, 0] -= centroid_x - target_width / 2
        resized_contour[:, 1] -= centroid_y - target_height / 2

        return resized_contour.astype(np.int32)

    canvas = np.zeros((200, 200, 3), dtype=np.uint8)

    # Calculate the maximum dimensions of the contours
    max_width = max(cv2.boundingRect(max_contour1)[2], cv2.boundingRect(max_contour2)[2])
    max_height = max(cv2.boundingRect(max_contour1)[3], cv2.boundingRect(max_contour2)[3])

    # Resize contours while maintaining aspect ratio and translate to center
    max_contour1_resized = resize_contour_with_aspect_ratio(max_contour1, max_width, max_height)
    max_contour2_resized = resize_contour_with_aspect_ratio(max_contour2, max_width, max_height)

    # Calculate the translation to center the contour within the bounding box
    contour1_x, contour1_y, contour1_w, contour1_h = cv2.boundingRect(max_contour1_resized)
    contour2_x, contour2_y, contour2_w, contour2_h = cv2.boundingRect(max_contour2_resized)

    translation1 = ((200 - contour1_w) // 2 - contour1_x, (200 - contour1_h) // 2 - contour1_y)
    translation2 = ((200 - contour2_w) // 2 - contour2_x, (200 - contour2_h) // 2 - contour2_y)

    # Draw contours on canvas
    cv2.drawContours(canvas, [max_contour1_resized + translation1], -1, (0, 255, 0), thickness=2)
    cv2.drawContours(canvas, [max_contour2_resized + translation2], -1, (255, 0, 0), thickness=2)

    # Add text denoting Teacher and Student
    cv2.putText(canvas, 'Teacher', (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Student', (120, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


    # Display the result
    cv2.imshow('Contours on Canvas', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
marking(r'C:\Users\Lenovo\Desktop\FYP\WriteRight\main\Images\Template writing\A.jpeg', r'C:\Users\Lenovo\Desktop\FYP\WriteRight\main\Images\Student writing\A.jpeg')