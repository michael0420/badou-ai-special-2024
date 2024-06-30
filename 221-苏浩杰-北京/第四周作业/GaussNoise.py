"""
手动实现高四噪声
"""

import cv2
import random

means = 8  # 均值
percetage = 1  # 设置噪声的像素点的数量
sigma =16  # 标准差


def GaussianNoise(src, channel=1):
    newImg = src.copy()
    for i in range(channel):
        newimg_c = newImg[:, :, i]
        newImgNum = int(newimg_c.shape[0] * newimg_c.shape[1] * percetage)
        for j in range(newImgNum):
            # 高斯噪声边缘值不处理
            x = random.randint(0, newimg_c.shape[0] - 1)
            y = random.randint(0, newimg_c.shape[1] - 1)

            newImg[x, y, i] = newimg_c[x, y] + random.gauss(means, sigma)
            if newImg[x, y, i] < 0:
                newImg[x, y, i] = 0
            elif newImg[x, y, i] > 255:
                newImg[x, y, i] = 255
    return newImg


src_gray = cv2.imread("../lenna.png")
_, _, channel = src_gray.shape
newImg = GaussianNoise(src_gray, channel)
# src = cv2.imread("../lenna.png")
# src_gray2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow("src2", src_gray)
cv2.imshow("gauss new image", newImg)
cv2.waitKey(0)
