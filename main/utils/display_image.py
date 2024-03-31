import cv2

def display_images(image1, image2, combined_images):
    image1_resized = cv2.resize(image1, (400, 400))
    image2_resized = cv2.resize(image2, (400, 400))
    combined_images_resized = cv2.resize(combined_images, (400, 400))

    cv2.imshow("Student writing", image2_resized)
    cv2.imshow("Teacher writing", image1_resized)
    cv2.imshow('End result', combined_images_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()