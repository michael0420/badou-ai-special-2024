# 实现椒盐噪声
import cv2
import numpy as np

def add_salt_and_pepper_noise(image, salt_ratio, pepper_ratio):
    # 复制原始图像，以免修改原始数据
    noisy_image = np.copy(image)
    # 计算需要添加噪声的像素数量
    num_salt = int(image.size * salt_ratio)
    num_pepper = int(image.size * pepper_ratio)
    # 添加盐噪声
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    # 添加椒噪声
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    # 返回添加椒盐噪声后的图片
    return noisy_image

# 读取灰度图像
image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
# 设置椒盐噪声的比例
salt_ratio = 0.1
pepper_ratio = 0.1
# 添加椒盐噪声
noisy_image = add_salt_and_pepper_noise(image, salt_ratio, pepper_ratio)
# 显示原始图像
cv2.imshow('Original Image', image)
# 显示添加椒盐噪声后的图像
cv2.imshow('Noisy Image', noisy_image)
# 等待用户按下任意键退出
cv2.waitKey(0)