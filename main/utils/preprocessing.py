import cv2

def preprocess(image_path):
    try:
        # Open an image file
        image = cv2.imread(image_path)
        
        if image is None:
            print("Error: Unable to open image file")
            return None
        
        # Resize image
        image = cv2.resize(image, (800, 800))
      
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold to binarize the image
        _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

        # Apply median filter to remove noise
        median_filtered_image = cv2.medianBlur(binary_image, 5)  # Adjust the kernel size as needed

        # Compute Laplacian of the median-filtered image
        laplacian_image = cv2.Laplacian(median_filtered_image, cv2.CV_64F)

        # Invert Laplacian image for display
        inverted_laplacian_img = 255 - cv2.convertScaleAbs(laplacian_image)

        # Apply thresholding to create binary mask
        _, laplacian_mask = cv2.threshold(inverted_laplacian_img, 50, 255, cv2.THRESH_BINARY)

        # Apply mask to the median-blurred image
        laplacian_enhanced_image = cv2.bitwise_and(median_filtered_image, median_filtered_image, mask=laplacian_mask)


        laplacian_enhanced_image = cv2.resize(laplacian_enhanced_image, (400, 400))

        # Display the enhanced image
        # cv2.imshow("Laplacian Enhanced Image", laplacian_enhanced_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return laplacian_enhanced_image
    
    except Exception as e:
        print("Error during image preprocessing:", str(e))
        return None

