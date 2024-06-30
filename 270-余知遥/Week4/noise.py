import cv2
import numpy as np

def salt_pepper_noise(image, salt_prob, pepper_prob):
    # 获取图像数据类型
    img_type = image.dtype
    # 计算椒盐噪声点数
    total_pixels = image.shape[0]*image.shape[1]
    num_salt = total_pixels * salt_prob
    num_pepper = total_pixels * pepper_prob
    # 添加椒盐噪声
    for i in range(int(num_salt)):
        x_coord=np.random.randint(0, image.shape[1] - 1)
        y_coord=np.random.randint(0, image.shape[0] - 1)
        image[y_coord,x_coord] = 255
    for i in range(int(num_pepper)):
        x_coord=np.random.randint(0, image.shape[1] - 1)
        y_coord=np.random.randint(0, image.shape[0] - 1)
        image[y_coord,x_coord] = 0
    return image.astype(img_type)

def gaussian_noise(image, mean, stddev):
    # 获取图像数据类型
    img_type = image.dtype
    # 添加高斯噪声
    noise = np.random.normal(mean, stddev, image.shape)
    noisy_image = image + noise
    return noisy_image.astype(img_type)

# 读取图像
img = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original img', img)
# 添加椒盐噪声
salt_pepper_img = salt_pepper_noise(img.copy(), 0.1, 0.1)
cv2.imshow('Salt and Pepper Noise', salt_pepper_img)
# 添加高斯噪声
gaussian_img = gaussian_noise(img.copy(), 0, 1)
cv2.imshow('Gaussian Noise', gaussian_img)

cv2.waitKey(0)