# 实现sobel边缘检测
import cv2

# 读取输入图像
input_image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
# 对图像进行 Sobel 边缘检测
sobel_x = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=3)  # x 方向
sobel_y = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=3)  # y 方向
# 将结果转换为 uint8 形式
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
# 合并 x 和 y 方向的 Sobel 输出
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
# 显示结果
cv2.imshow('Input Image', input_image)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.imshow('Sobel Combined', sobel_combined)
# 等待用户按下任意键
cv2.waitKey(0)
