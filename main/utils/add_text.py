import cv2


def add_text(image2, similarity_grade):
    if image2 is None or image2.size == 0:
        print("Error: Invalid or empty image2")
        return None

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_color = (0, 255, 0)
    thickness = 2
    line_type = 2
    text = f"Overall Similarity: {similarity_grade:.2f}"

    # cv2.putText(image2, text, (10, 110), font, font_scale, font_color, thickness, line_type)
    
    return image2