import cv2

# 加载图像
img = cv2.imread('D:/AI--pan/code/lenna.png')

# 将图像转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行直方图均衡化
equalized_img = cv2.equalizeHist(gray_img)

# 显示原始图像和直方图均衡化后的图像
cv2.imshow('Original Image', gray_img)
cv2.imshow('Equalized Image', equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
