import cv2
import numpy as np


import cv2
import numpy as np

def marking(image_path1, image_path2):
    try:
        # Read images
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)

        if image1 is None:
            raise ValueError(f"Image at {image_path1} could not be read.")
        if image2 is None:
            raise ValueError(f"Image at {image_path2} could not be read.")

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        edges1 = cv2.Canny(gray1, 100, 200)
        edges2 = cv2.Canny(gray2, 100, 200)

        edge_density_threshold = 0.001  # Adjust as needed

        total_pixels1 = gray1.shape[0] * gray1.shape[1]
        total_pixels2 = gray2.shape[0] * gray2.shape[1]
        edge_pixels1 = np.count_nonzero(edges1)
        edge_pixels2 = np.count_nonzero(edges2)
        edge_density1 = edge_pixels1 / total_pixels1
        edge_density2 = edge_pixels2 / total_pixels2

        print(edge_density1)
        print(edge_density2)

        if edge_density1 > edge_density_threshold:
            kernel1 = np.ones((1, 1), np.uint8)
            gray1 = cv2.dilate(gray1, kernel1, iterations=1)
            edges1 = cv2.Canny(gray1, 50, 200)
        else:
            kernel1 = np.ones((5, 5), np.uint8)
            gray1 = cv2.dilate(gray1, kernel1, iterations=1)

        contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours1 or not contours2:
            raise ValueError("No contours found in one or both images.")

        max_contour1 = max(contours1, key=cv2.contourArea)
        max_contour2 = max(contours2, key=cv2.contourArea)

        def resize_contour_with_aspect_ratio(contour, target_width, target_height):
            if contour.ndim == 3:
                contour = contour.squeeze(1)

            x, y, w, h = cv2.boundingRect(contour)

            if w == 0 or h == 0:
                raise ValueError("Width or height of the bounding rectangle is zero.")

            scale_factor_w = target_width / w
            scale_factor_h = target_height / h
            scale_factor = min(scale_factor_w, scale_factor_h)

            resized_contour = contour * scale_factor

            centroid_x = (x + w / 2)
            centroid_y = (y + h / 2)
            resized_contour[:, 0] -= centroid_x - target_width / 2
            resized_contour[:, 1] -= centroid_y - target_height / 2

            return resized_contour.astype(np.int32)

        canvas = np.zeros((400, 400, 3), dtype=np.uint8)

        max_width = max(cv2.boundingRect(max_contour1)[2], cv2.boundingRect(max_contour2)[2])
        max_height = max(cv2.boundingRect(max_contour1)[3], cv2.boundingRect(max_contour2)[3])

        max_contour1_resized = resize_contour_with_aspect_ratio(max_contour1, max_width, max_height)
        max_contour2_resized = resize_contour_with_aspect_ratio(max_contour2, max_width, max_height)

        contour1_x, contour1_y, _, _ = cv2.boundingRect(max_contour1_resized)
        contour2_x, contour2_y, _, _ = cv2.boundingRect(max_contour2_resized)

        # Calculate translation to center horizontally
        translation1 = (100 - contour1_x, 100 - contour1_y)
        translation2 = (100 - contour2_x, 100 - contour2_y)

        cv2.drawContours(canvas, [max_contour1_resized + translation1], -1, (0, 255, 0), thickness=2)
        cv2.drawContours(canvas, [max_contour2_resized + translation2], -1, (255, 0, 0), thickness=2)
        
        # cv2.imshow('Contours on Canvas', canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return canvas

    except Exception as e:
        print("An error occurred in the marking function:", str(e))
        return None
