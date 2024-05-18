import cv2

# 读取图像（灰度图像）
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 使用 Sobel 算子进行边缘检测
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

absX = cv2.convertScaleAbs(sobel_x)
absY = cv2.convertScaleAbs(sobel_y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# 显示结果
cv2.imshow('Original Image', img)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', absY)
cv2.imshow('Sobel X+Y', dst)
cv2.waitKey(5000)
cv2.destroyAllWindows()
