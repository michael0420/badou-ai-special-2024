# 实现直方图均衡化
import cv2

# 灰度图像直方图均衡化
# 读取输入图像并灰度化
input_image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
# 进行直方图均衡化
output_image = cv2.equalizeHist(input_image)
# 显示结果
cv2.imshow('Input Image', input_image)
cv2.imshow('Equalized Image', output_image)


# 彩色图像直方图均衡化
# 读取输入图像
input_color_image = cv2.imread('lenna.png')
# 分解通道
b, g, r = cv2.split(input_color_image)
# 对每个通道进行直方图均衡化
b_eq = cv2.equalizeHist(b)
g_eq = cv2.equalizeHist(g)
r_eq = cv2.equalizeHist(r)
# 合并通道
output_color_image = cv2.merge((b_eq, g_eq, r_eq))
# 显示结果
cv2.imshow('Input Color Image', input_color_image)
cv2.imshow('Equalized Color Image', output_color_image)
# 等待用户按下任意键
cv2.waitKey(0)