import cv2
import numpy as np
import random

def gaussian_noise(img, mu, sigma, percentage):
    noise_img = img.copy()
    rows, cols, _ = noise_img.shape # 获取图像的高和宽
    noise_num = int(percentage * rows * cols)  # 要添加的噪声点的数量
    for i in range(noise_num):
        # 随机选出噪点位置坐标
        randX = random.randint(0, rows - 1)
        randY = random.randint(0, cols - 1)
        # 添加高斯噪声
        noise = random.gauss(mu, sigma)
        # 将噪声值加到像素值上
        noise_img[randX, randY] = np.clip(noise_img[randX, randY] + noise, 0, 255).astype(np.uint8)

    return noise_img

img = cv2.imread('face.jpg')
noise_image = gaussian_noise(img, 20, 4, 0.6)

cv2.imshow('src', img)
cv2.imshow('noise', noise_image)
cv2.waitKey(0)
cv2.destroyAllWindows()