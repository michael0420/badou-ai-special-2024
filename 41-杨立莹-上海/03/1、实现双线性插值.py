# 实现双线性插值
import cv2
import numpy as np

def bilinear_interpolation(image, output_width, output_height):
    # 获取图像尺寸和通道数
    height, width, channels = image.shape
    # 创建一个新的图像来存储插值结果
    interpolation_image = np.zeros((output_height, output_width, channels), dtype=np.uint8)
    # 计算缩放比例
    y_ratio = height / output_height  # 计算纵向缩放比例
    x_ratio = width / output_width  # 计算横向缩放比例
    # 循环遍历新图像的每个像素
    for h in range(output_height):
        for w in range(output_width):
            # 计算在原始图像中对应的位置，加上0.5是为了保持中心对称
            src_w = w * x_ratio + 0.5
            src_h = h * y_ratio + 0.5

            # 边界检查，确保计算的位置不会超出原始图像的边界
            src_w = max(0, min(src_w, width - 1))
            src_h = max(0, min(src_h, height - 1))

            # 找到最近的四个像素，用于双线性插值
            x1 = int(src_w)  # 左上角像素的x坐标
            y1 = int(src_h)  # 左上角像素的y坐标
            x2 = min(x1 + 1, width - 1)  # 右上角像素的x坐标
            y2 = min(y1 + 1, height - 1)  # 左下角像素的y坐标

            # 计算插值权重
            dx = src_w - x1  # x方向的插值权重
            dy = src_h - y1  # y方向的插值权重

            # 进行双线性插值，对图像的每个通道进行插值并进行范围限制
            for c in range(channels):  # 遍历图像的每个通道
                interpolation_image[h, w, c] = np.clip(
                    (1 - dx) * (1 - dy) * image[y1, x1, c] +  # 左上角像素的插值
                    dx * (1 - dy) * image[y1, x2, c] +  # 右上角像素的插值
                    (1 - dx) * dy * image[y2, x1, c] +  # 左下角像素的插值
                    dx * dy * image[y2, x2, c],  # 右下角像素的插值
                    0, 255  # 范围限制，确保插值结果在[0, 255]范围内
                )
    # 返回双线性插值后的图片
    return interpolation_image

# 读取彩色图像
color_img = cv2.imread('lenna.png')
# 设置输出图像的宽度和高度
output_width = 800
output_height = 800
# 进行双线性插值
interpolation_image = bilinear_interpolation(color_img, output_width, output_height)
# 显示双线性插值结果
cv2.imshow('Interpolation Image', interpolation_image)
# 等待用户按下任意键
cv2.waitKey(0)