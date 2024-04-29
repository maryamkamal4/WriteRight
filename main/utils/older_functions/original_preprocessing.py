import cv2


# Open an image file
image_path = r"C:\Users\Lenovo\Desktop\FYP\WriteRight\main\Images\Template writing\level1.jpeg"  # Change this to the path of your image file
image = cv2.imread(image_path)
image = cv2.resize(image, (800,800))

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold to binarize the image
_, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

# Apply median filter to remove noise
median_filtered_image = cv2.medianBlur(binary_image, 5)  # Adjust the kernel size as needed

# Compute Laplacian of the median-filtered image
laplacian_image = cv2.Laplacian(median_filtered_image, cv2.CV_64F)

# Compute gradient (first-order derivative) using Sobel operator
# gradient_x = cv2.Sobel(median_filtered_image, cv2.CV_64F, 1, 0, ksize=3)
# gradient_y = cv2.Sobel(median_filtered_image, cv2.CV_64F, 0, 1, ksize=3)
# gradient_image = cv2.magnitude(gradient_x, gradient_y)

# Invert Laplacian and gradient images for display
inverted_laplacian_img = 255 - cv2.convertScaleAbs(laplacian_image)
# inverted_gradient_img = 255 - cv2.convertScaleAbs(gradient_image)

# Apply thresholding to create binary masks
_, laplacian_mask = cv2.threshold(inverted_laplacian_img, 50, 255, cv2.THRESH_BINARY)
# _, gradient_mask = cv2.threshold(inverted_gradient_img, 50, 255, cv2.THRESH_BINARY)

# Apply masks to the median-blurred image separately
laplacian_enhanced_image = cv2.bitwise_and(median_filtered_image, median_filtered_image, mask=laplacian_mask)
# gradient_enhanced_image = cv2.bitwise_and(median_filtered_image, median_filtered_image, mask=gradient_mask)

# Display the enhanced images
cv2.imshow("Laplacian Enhanced Image", laplacian_enhanced_image)
# cv2.imshow("Gradient Enhanced Image", gradient_enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()