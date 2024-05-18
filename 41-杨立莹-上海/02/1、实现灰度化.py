# 实现灰度化
import cv2
import numpy as np

def rgb_to_gray(rgb_img):
    # 获取图像的尺寸
    height, width, channels = rgb_img.shape
    # 创建一个空的灰度图像
    gray_img = np.zeros((height, width), np.uint8)
    # 循环遍历图像的每个像素，并计算灰度值
    for h in range(height):
        for w in range(width):
            # 从RGB图像中获取像素值
            r, g, b = rgb_img[h, w]
            # 计算灰度值
            gray_value = r * 0.3 + g * 0.59 + b * 0.11
            # 将灰度值保存到灰度图像中
            gray_img[h, w] = int(gray_value)
    # 返回灰度图像
    return gray_img

# 读取彩色图像
color_img = cv2.imread('lenna.png')
# 将彩色图像转换为灰度图像
gray_img = rgb_to_gray(color_img)
# 显示灰度图像
cv2.imshow('Gray Image', gray_img)
# 等待用户按下任意键
cv2.waitKey(0)