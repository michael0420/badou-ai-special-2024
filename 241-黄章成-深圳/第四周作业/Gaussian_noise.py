#随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
import numpy as np
import cv2
from numpy import shape
import random

# means: 高斯噪声的均值，决定了噪声的强度
# sigma: 高斯噪声的方差，决定了噪声的范围
# percentage: 噪声占图像的比例。这应当是0到1之间的一个数字，表示希望多少比例的图像像素添加噪声。
def GaussianNoise(src, means, sigma, percentage):
    NoiseImg = src
    NoiseNum = int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
		# 每次取一个随机点
		# 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
		# 高斯噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # 此处在原有像素灰度值上加上随机数
        NoiseImg[randX, randY] = NoiseImg[randX, randY]+random.gauss(means, sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg
img = cv2.imread('lenna.png', 0)
img1 = GaussianNoise(img, 2, 4, 0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source', img2)
cv2.imshow('lenna_GaussianNoise', img1)
cv2.waitKey(0)