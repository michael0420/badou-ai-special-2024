import cv2

# 读取图像
image_path = "D:/AI--pan/code/lenna.png"
image = cv2.imread(image_path)

# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 对灰度图进行二值化处理
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 将三张图像水平拼接在一起
combined_image = cv2.hconcat([image, cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)])

# 显示合并后的图像
cv2.imshow("Combined Image", combined_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
