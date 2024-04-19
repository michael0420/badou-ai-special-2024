'''
作业1：实现高斯噪声
作业2：实现椒盐噪声
'''

import numpy as np
import cv2
import random
from skimage import util


# 定义一个函数，实现高斯噪声
def gauss_noise(img, mean, sigma, k):

    img_noise = img.copy()
    # 根据添加噪声的比例，以及图像大小计算循环次数
    noiseNum = int(img_noise.shape[0] * img_noise.shape[1] * k)

    # 循环随机给某个像素点添加高斯噪声
    for i in range(noiseNum):
        # 随机取像素点的位置信息
        randX = random.randint(0, img_noise.shape[0] - 1)
        randY = random.randint(0, img_noise.shape[1] - 1)

        # 给像素点添加高斯噪声
        img_noise[randX, randY] += random.gauss(mean, sigma)

        if img_noise[randX, randY] > 255:
            img_noise[randX, randY] = 255
        elif img_noise[randX, randY] < 0:
            img_noise[randX, randY] = 0

    return img_noise


# 定义一个函数实现椒盐噪声
def salt_and_pepper(img, k):

    img_noise = img.copy()
    # 根据添加噪声的比例，以及图像大小计算循环次数
    noiseNum = int(img_noise.shape[0] * img_noise.shape[1] * k)

    # 循环随机给某个像素点添加椒盐噪声
    for i in range(noiseNum):
        # 随机取像素点的位置信息
        randX = random.randint(0, img_noise.shape[0] - 1)
        randY = random.randint(0, img_noise.shape[1] - 1)

        # 给像素点添加椒盐噪声
        if random.randint(1, 2) == 1:
            img_noise[randX, randY] = 255
        else:
            img_noise[randX, randY] = 0

    return img_noise
    pass


if __name__ == '__main__':
    # 以灰度图的方式读取一个图片：
    img_gray = cv2.imread('lenna.png', 0)
    img_gauss = gauss_noise(img_gray, 2, 4, 0.8)
    img_salt = salt_and_pepper(img_gray, 0.5)

#    img_gray = cv2.imread('lenna.png', 0)
    # 展示原图和加噪后的图像
    print(img_gray, img_gauss, img_salt)
    cv2.imshow('img', np.hstack([img_gray, img_gauss, img_salt]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 使用已有库进行加噪
    img_poisson = util.random_noise(img_gray, mode='poisson')
    print(img_gray)
#    cv2.imshow('img1', np.hstack([img_gray, img_poisson]))
    cv2.imshow('img2', img_poisson)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
