import numpy as np
import cv2
from numpy import shape
import random


def fun1(src, percetage):
    NoiseImg = src    # 可以指定区域进行加噪
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])  # 需要加噪的像素数目 NP=SP(总数)*SNR(信噪比)
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        # 【不同的噪声产生逻辑】
        # random.random()用于生成一个0到1的随机符点数: 0 <= n < 1.0  例如: 0.3558774735558118
        # random.random生成随机浮点数，0.5->随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0      # 椒噪声(黑色0)
        else:
            NoiseImg[randX, randY] = 255    # 盐噪声(白色255)

    return NoiseImg

img = cv2.imread('D:\cv_workspace\picture\lenna.png', 0)  # 读入灰度图片
img1 = fun1(img, 0.2)  # 0.2 信噪比:信号和噪声所占比例 在0-1之间  因为椒盐噪声对比度比较明显所以0.2的效果也很明显
# 在文件夹中写入命名为lenna_PepperandSalt.png的加噪后的图片
# cv2.imwrite('lenna_PepperandSalt.png',img1)

img = cv2.imread('D:\cv_workspace\picture\lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', img2)
cv2.imshow('lenna_PepperandSalt', img1)
cv2.waitKey(0)
