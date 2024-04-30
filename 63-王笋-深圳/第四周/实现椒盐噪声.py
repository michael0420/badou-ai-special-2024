import numpy as np
import cv2
from numpy import shape
import random

# 对于图像来说：
# image.shape[0]——图片高
# image.shape[1]——图片长
# image.shape[2]——图片通道数
# 而对于矩阵来说：
# shape[0]：表示矩阵的行数
# shape[1]：表示矩阵的列数
def PepperSaltNoise(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])  # 指定信噪比
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生产的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 椒盐噪声图片边缘不处理，故-1
        randX = random.randrange(0, src.shape[0]-1)
        randY = random.randrange(0, src.shape[1]-1)
        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg

img = cv2.imread('lenna.png', 0)
img1 = PepperSaltNoise(img,0.2)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('lenna_GaussionNoise.jpg', img2)
cv2.imshow('source', img2)
cv2.imshow('lenna_PepperSalt',img1)
cv2.waitKey(0)