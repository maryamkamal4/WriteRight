from .calculate_similarity import calculate_similarity
from .find_differences import find_differences


def compare_characters(char1, char2, box1, box2, image1, height1, image2, height2, similarities, combined_imgs):
    if char1 == char2:
        box1 = box1.split(" ")
        box2 = box2.split(" ")

        x1_min, y1_min, x1_max, y1_max = int(box1[1]), int(box1[2]), int(box1[3]), int(box1[4])
        x2_min, y2_min, x2_max, y2_max = int(box2[1]), int(box2[2]), int(box2[3]), int(box2[4])

        region_of_interest1 = image1[height1 - y1_max:height1 - y1_min, x1_min:x1_max]
        region_of_interest2 = image2[height2 - y2_max:height2 - y2_min, x2_min:x2_max]

        similarity_grade = calculate_similarity(region_of_interest1, region_of_interest2)
        similarities.append(similarity_grade)

        combined_img = find_differences(region_of_interest1, region_of_interest2)
        combined_imgs.append(combined_img)
    else:
        similarity_grade = 0.0
        similarities.append(similarity_grade)

    print("Similarity Grade: {:.2f}".format(similarity_grade))