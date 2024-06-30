# 实现二值化
import cv2
import numpy as np

def rgb_to_gray(rgb_img):
    # 转换为灰度图像
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    return gray_img

def binary_threshold(gray_img, threshold):
    # 创建一个空白的二值化图像
    height, width = gray_img.shape
    binary_img = np.zeros((height, width), np.uint8)
    # 循环遍历图像的每个像素，并进行二值化处理
    for h in range(height):
        for w in range(width):
            # 获取当前像素的灰度值
            pixel_value = gray_img[h, w]
            # 根据阈值进行二值化处理
            if pixel_value >= threshold:
                binary_img[h, w] = 255
            else:
                binary_img[h, w] = 0
    # 返回二值化图像
    return binary_img

# 读取彩色图像
color_img = cv2.imread('lenna.png')
# 将彩色图像转换为灰度图像
gray_img = rgb_to_gray(color_img)
# 设置阈值
threshold_value = 127
# 对灰度图像进行二值化处理
binary_img = binary_threshold(gray_img, threshold_value)
# 显示二值化图像
cv2.imshow('Binary Image', binary_img)
# 等待用户按下任意键
cv2.waitKey(0)