import cv2
import numpy as np
import random

def add_gaussian_noise(image, mean, sigma, percentage):
    # 复制原始图像，以免修改原始数据
    noisy_image = np.copy(image)
    # 计算需要添加噪声的像素数量
    num_noise_pixels = int(percentage * image.size)
    # 循环添加噪声
    for i in range(num_noise_pixels):
        # 随机选择一个像素位置
        x = random.randint(0, image.shape[0] - 1)
        y = random.randint(0, image.shape[1] - 1)
        # 生成高斯噪声
        noise = random.gauss(mean, sigma)
        # 添加噪声到像素值
        noisy_image[x, y] += noise
        # 确保像素值在0到255之间
        noisy_image[x, y] = np.clip(noisy_image[x, y], 0, 255)
    # 返回添加高斯噪声后的图片
    return noisy_image.astype(np.uint8)

# 读取灰度图像
image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
# 设置添加噪声的参数
mean = 0
sigma = 25
percentage = 0.8
# 添加高斯噪声
noisy_image = add_gaussian_noise(image, mean, sigma, percentage)
# 显示原始图像
cv2.imshow('Original Image', image)
# 显示添加噪声后的图像
cv2.imshow('Noisy Image', noisy_image)
# 等待用户按下任意键
cv2.waitKey(0)