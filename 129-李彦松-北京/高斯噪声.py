import numpy as np
import cv2
from skimage import util

def GaussianNoise_optimized(src, means, sigma, percentage):
    # 将图像转换为 float64 类型
    NoiseImg = src.astype(np.float64)

    total_pixels = src.shape[0] * src.shape[1]
    NoiseNum = int(percentage * total_pixels)

    # 生成噪声的随机坐标
    randX = np.random.randint(0, src.shape[0], size=NoiseNum)
    randY = np.random.randint(0, src.shape[1], size=NoiseNum)

    # 生成高斯噪声
    noise = np.random.normal(means, sigma, size=NoiseNum)

    # 检查噪声是否会导致像素溢出/下溢
    noise = np.where(NoiseImg[randX, randY] + noise < 255, noise, 255 - NoiseImg[randX, randY])
    noise = np.where(NoiseImg[randX, randY] + noise > 0, noise, -NoiseImg[randX, randY])

    # 将噪声添加到图像上
    NoiseImg[randX, randY] += noise

    # 将图像转换回 uint8 类型
    NoiseImg = NoiseImg.astype(np.uint8)

    return NoiseImg

img = cv2.imread('lenna.png',0)
cv2.imshow('source',img)
img1 = GaussianNoise_optimized(img,2,4,0.8)
cv2.imshow('lenna_GaussianNoise', img1)
# 使用 skimage 库中的 random_noise 函数添加高斯噪声
img = cv2.imread('lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
noise_gs_img=util.random_noise(img,mode='gaussian')
cv2.imshow("lenna_",noise_gs_img)

cv2.waitKey(0)