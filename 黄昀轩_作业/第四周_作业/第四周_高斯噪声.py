"""
@author: huangyunxuan

高斯噪声

"""
import random

import cv2


def G_Noise(src, mu, sigma, percentage):
    NoiseImg = src
    NoiseNums = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNums):
        rX = random.randint(0, src.shape[0] - 1)
        rY = random.randint(0, src.shape[1] - 1)
        NoiseImg[rX, rY] += random.gauss(mu, sigma)
        if NoiseImg[rX, rY] < 0:
            NoiseImg[rX, rY] = 0
        elif NoiseImg[rX, rY] > 255:
            NoiseImg[rX, rY] = 255
    return NoiseImg


img = cv2.imread('lenna.png', 0)
dst = G_Noise(img, 0, 1, 0.7)
cv2.imshow("G_noise", dst)

# 调用skimage模块
# img2 = cv2.imread('lenna.png',0)
# noise_gs_img=util.random_noise(img2,mode="gaussian")
# cv2.imshow("noise_gs_img",noise_gs_img)
#
cv2.waitKey(0)
