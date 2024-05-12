import cv2


def add_text(image2, similarity_grade, boxes2, height2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_color = (0, 255, 0)
    thickness = 2
    line_type = 2
    text = f"Overall Similarity: {similarity_grade:.2f}"

    cv2.putText(image2, text, (10, 110), font, font_scale, font_color, thickness, line_type)
    
    cv2.imshow('Textual Image', image2)
    cv2.waitKey(0)
    
    return image2