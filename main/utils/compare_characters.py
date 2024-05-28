from matplotlib import pyplot as plt
from .calculate_similarity import calculate_similarity


def compare_characters(char1, char2, box1, box2, image1, height1, image2, height2):
    try:
        # Split the bounding boxes into individual coordinates
        box1 = box1.split(" ")
        box2 = box2.split(" ")

        x1_min, y1_min, x1_max, y1_max = int(box1[1]), int(box1[2]), int(box1[3]), int(box1[4])
        x2_min, y2_min, x2_max, y2_max = int(box2[1]), int(box2[2]), int(box2[3]), int(box2[4])

        # Extract regions of interest based on bounding boxes
        region_of_interest1 = image1[height1 - y1_max:height1 - y1_min, x1_min:x1_max]
        region_of_interest2 = image2[height2 - y2_max:height2 - y2_min, x2_min:x2_max]

        # Calculate similarity grade
        similarity_grade = calculate_similarity(region_of_interest1, region_of_interest2)

        # If characters are the same, use the similarity grade as is
        # If characters are different, adjust the similarity grade
        if char1 != char2:
            similarity_grade /= 2
            
        similarity_grade = round(similarity_grade, 2)

        print("Similarity Grade: {:.2f}".format(similarity_grade))
        return similarity_grade

    except Exception as e:
        print("Error during character comparison:", str(e))
        return None
