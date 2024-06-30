# coding = utf-8

'''
    实现高斯噪声
'''

import numpy as np
import  cv2
import random
from skimage import util as ski


img = cv2.imread('lenna.png')
cv2.imshow('lenna', img)


'''
使用接口实现高斯噪声
'''
# gaussImg = ski.random_noise(img, mode='gaussian')
# cv2.imshow('gauss', gaussImg)


'''
手动实现高斯噪声
'''
def gauss(img, mean, sigma, perc):
    noise_img = img
    noise_num = int(img.shape[0] * img.shape[1] * perc)
    # print(np.shape(noise_img))

    # # 遍历3通道
    # for j in range(noise_img.shape[2]):
    #     # 遍历噪声点出现的随机坐标
    #     for i in range(noise_num):
    #         # 每次取一个随机点
    #         # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
    #         # random.randint()生成随机整数
    #         # 高斯噪声图片边缘不处理，故-1
    #         randX = random.randint(0, img.shape[0] - 1)
    #         randY = random.randint(0, img.shape[1] - 1)
    #         # 在原有像素像素值上加上随机数
    #         noise_img[randX, randY, j] = noise_img[randX, randY, j] + random.gauss(mean, sigma)
    #
    #         # 约束像素值在[0,255]之间
    #         if noise_img[randX, randY, j] < 0:
    #             noise_img[randX, randY, j] = 0
    #         elif noise_img[randX, randY, j] > 255:
    #             noise_img[randX, randY, j] = 255


    indices = set()
    while len(indices) < noise_num:
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        indices.add((randX, randY))

        # 遍历所有通道并添加噪声
    for c in range(img.shape[2]):
        for x, y in indices:
            # np.random.normal用于从正态分布中抽取随机样本
            noise_value = np.random.normal(mean, sigma)
            # 使用 clip() 函数限制值在 0 和 255 之间
            # 使用 .astype(np.uint8) 方法将其转换为 uint8 类型。注意，在转换过程中，任何超出 uint8 范围的值（即大于 255 的值）都会被截断到 255
            noise_img[x, y, c] = np.clip(noise_img[x, y, c] + noise_value, 0, 255).astype(np.uint8)
    return noise_img


src = cv2.imread('lenna.png')
gauss_noise = gauss(src, 3, 4, 0.7)
cv2.imshow('gaussImg', gauss_noise)
cv2.waitKey(0)
