import cv2
import numpy as np

# 读取图片
image = cv2.imread('lenna.png')

# 高斯噪声的参数
mean = 1  # 平均值
std = 2  # 标准差

# 创建高斯噪声
gaussian_noise = np.random.normal(mean, std, image.shape).astype('uint8')

# 将高斯噪声添加到图片上
noisy_image = cv2.add(image, gaussian_noise)

# 显示图片
cv2.imshow('Original Image', image)
# cv2.imshow('Gaussian Noise', gaussian_noise)
cv2.imshow('Noisy Image', noisy_image)

# 等待按键，然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()