# 随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
import numpy as np
import cv2
from numpy import shape
import random


def GaussianNoise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])  # 多少个点需要处理 percetage * H * W  percetage信噪比SNR
    '''
    range() 是Python的一个内置函数，返回的是一个可迭代对象。用于创建数字序列。
            range(start, stop, step) -> range(初值, 终值, 步长)
            比如：range(6)  返回从0到6（不包括6）的一系列数字范围，步长为1，如下所示：0,1,2,3,4,5
    random.randint生成随机整数
    '''
    for i in range(NoiseNum):  # 循环处理NoiseNum个像素点
        # 每次取一个随机点: NoiseImg[randX, randY]
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # 高斯噪声图片边缘不处理，故-1 [一个点对图像的影响很小] ; 重复的点影响也很小
        randX = random.randint(0, src.shape[0] - 1)  # 在 0 - H 之间随机取一个数
        randY = random.randint(0, src.shape[1] - 1)  # 在 0 - W 之间随机取一个数
        # 此处在原有像素灰度值上加上随机数
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255 (避免越界)
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


# 0：灰度模式（Grayscale），图像以灰度形式加载，每个像素的值在 0 到 255 之间。
img = cv2.imread('D:\cv_workspace\picture\lenna.png', 0)
img1 = GaussianNoise(img, 2, 4, 0.8)  # 2->means 4->sigma 0.8 给多少像素加噪的百分比
img = cv2.imread('D:\cv_workspace\picture\lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source', img2)
cv2.imshow('lenna_GaussianNoise', img1)
cv2.waitKey(0)
