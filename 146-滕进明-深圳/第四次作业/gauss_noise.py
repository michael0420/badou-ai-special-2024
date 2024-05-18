# 高斯噪声
import cv2
import numpy as np
import random
from numpy import shape

# 定义高斯噪声函数
def gauss_noise(scr, means, sigma, percentage):  # 随机生成符合正态(高斯)分布的随机数，means，sigma为两个参数
    img = scr
    random_range = int(percentage * scr.shape[0] * scr.shape[1])  # 确定需要被高斯噪声处理的图形像素数目
    for i in range(random_range):
        random_x = random.randint(0, scr.shape[0] - 1) # 高斯噪声化默认不处理边缘像素，因此有 -1 操作
        random_y = random.randint(0, scr.shape[1] - 1)

        # 限定范围内的随机像素高斯噪声处理
        img[random_x, random_y] = img[random_x, random_y] + random.gauss(means, sigma)

        # 处理像素值超过[0 ~ 255]取值范围的像素
        if img[random_x, random_y] > 255:
            img[random_x, random_y] = 255
        elif img[random_x, random_y] < 0:
            img[random_x, random_y] = 0
    return img

img = cv2.imread("lenna.png", 0) # flags参数用于指定读取图像的方式和格式。当flags参数为0时，表示以灰度图像的方式读取图像。
gaussian_img = gauss_noise(img, 2, 4, 0.8)

img2 = cv2.imread("lenna.png")
gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray_img", gray_img)
cv2.imshow("Gaussian Noise", gaussian_img)
cv2.waitKey(0)

