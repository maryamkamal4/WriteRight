# def find_differences(img1, img2):

#     print("*****original image in find diff*****")
#     print(img1.shape)
#     print(img2.shape)

#     img1 = cv2.resize(img1, (200,200))
#     img2 = cv2.resize(img2, (200,200))

#     print("*****resized image in find diff*****")
#     print(img1.shape)
#     print(img2.shape)

#     img1 = cv2.GaussianBlur(img1, (5, 5), 0)
#     img2 = cv2.GaussianBlur(img2, (5, 5), 0)

#     # Split the images into 4 quadrants
#     h, w = img1.shape[:2]
#     mid_h, mid_w = h // 2, w // 2
#     quadrants_img1 = [img1[:mid_h, :mid_w], img1[:mid_h, mid_w:], img1[mid_h:, :mid_w], img1[mid_h:, mid_w:]]
#     quadrants_img2 = [img2[:mid_h, :mid_w], img2[:mid_h, mid_w:], img2[mid_h:, :mid_w], img2[mid_h:, mid_w:]]

#     marked_quadrants = []

#     for quad1, quad2 in zip(quadrants_img1, quadrants_img2):
#         marked_quad = find_differences_in_quadrants(quad1, quad2)
#         marked_quadrants.append(marked_quad)

#     # Combine the marked quadrants into a single image
#     combined_image = np.vstack([np.hstack(marked_quadrants[:2]), np.hstack(marked_quadrants[2:])])

#     # Show the combined image
#     plt.imshow(combined_image, cmap='gray')
#     plt.title('Combined Image with Differences')
#     plt.axis('off')
#     plt.show()

#     return combined_image


# def find_differences_in_quadrants(quad1, quad2):
#     (score, diff) = ssim(quad1, quad2, full=True)

#     diff = (diff * 255).astype("uint8")
#     _, thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)

#     contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#     contours = [c for c in contours if 200 < cv2.contourArea(c) < 800]

#     marked_quad = cv2.cvtColor(quad1, cv2.COLOR_GRAY2BGR)

#     if len(contours):
#         for c in contours:
#             x, y, w, h = cv2.boundingRect(c)
#             cv2.rectangle(marked_quad, (x, y), (x + w, y + h), (0, 0, 255), 4)

#     return marked_quad