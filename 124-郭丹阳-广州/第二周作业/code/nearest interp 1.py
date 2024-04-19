import cv2

# 读取图像
image_path = "D:/AI--pan/code/lenna.png"
image = cv2.imread(image_path)

# 放大图像
scale_factor = 2  # 放大倍数
resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

# 缩小图像
scale_factor = 0.5  # 缩小倍数
resized_image_smaller = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

# 显示原始图像、放大图像和缩小图像
cv2.imshow("Original Image", image)
cv2.imshow("Resized Image (Larger)", resized_image)
cv2.imshow("Resized Image (Smaller)", resized_image_smaller)

cv2.waitKey(0)
cv2.destroyAllWindows()
