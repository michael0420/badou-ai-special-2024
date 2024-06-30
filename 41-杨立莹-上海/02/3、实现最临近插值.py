# 实现最临近插值
import cv2
import numpy as np

def nearest_neighbor_interpolation(image, new_height, new_width):
    # 获取彩色图像的高度、宽度和通道数
    height, width, channels = image.shape
    # 计算高度和宽度的缩放比例
    height_ratio = height / new_height
    width_ratio = width / new_width
    # 创建一个新的空白彩色图像，用于存储插值结果
    interpolated_image = np.zeros((new_height, new_width, channels), np.uint8)
    # 对新图像的每个像素进行插值计算
    for h in range(new_height):
        for w in range(new_width):
            # 计算原始图像中对应的位置
            original_h = int(h * height_ratio)
            original_w = int(w * width_ratio)
            # 对每个通道分别进行最邻近插值
            for c in range(channels):
                interpolated_image[h, w, c] = image[original_h, original_w, c]
    # 返回插值后的图片
    return interpolated_image

# 读取彩色图像
color_img = cv2.imread('lenna.png')
# 设置新图像的尺寸
new_height = 800
new_width = 800
# 进行最邻近插值
interpolated_image = nearest_neighbor_interpolation(color_img, new_height, new_width)
# 显示插值结果
cv2.imshow('Interpolated Image', interpolated_image)
# 等待用户按下任意键
cv2.waitKey(0)