import cv2


def add_text(image2, overall_similarity, similarities, boxes2, height2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_color = (0, 255, 0)
    thickness = 2
    line_type = 2
    text = f"Overall Similarity: {overall_similarity:.2f}"

    cv2.putText(image2, text, (10, 110), font, font_scale, font_color, thickness, line_type)

    for similarity_grade, box2 in zip(similarities, boxes2.splitlines()):
        if similarity_grade == 0.0:
            box2 = box2.split(" ")
            x2_min, y2_min, x2_max, y2_max = int(box2[1]), int(box2[2]), int(box2[3]), int(box2[4])
            cv2.rectangle(image2, (x2_min, height2 - y2_max), (x2_max, height2 - y2_min), (0, 0, 255), 2)
    
    cv2.imshow('Resized Image 2', image2)
    cv2.waitKey(0)
    
    return image2