
import cv2
import numpy as np

image_path = 'D:/AI--pan/code/lenna.png'
image = cv2.imread(image_path)

# 生成高斯噪声
mean = 0
std_dev = 25  # 标准差，控制噪声的强度
gaussian_noise = np.zeros_like(image, np.uint8)
cv2.randn(gaussian_noise, mean, std_dev)

# 添加高斯噪声到图像
noisy_image = cv2.add(image, gaussian_noise)

# 显示图像
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
