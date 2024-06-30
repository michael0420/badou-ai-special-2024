"""
@author: huangyunxuan

椒盐噪声

"""
import random

import cv2
from skimage import util


def func(src, percentage):
    NoiseImg = src
    NoiseNums = int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNums):
        rX = random.randint(0, src.shape[0] - 1)
        rY = random.randint(0, src.shape[1] - 1)
        if random.random() < 0.5:
            NoiseImg[rX, rY] = 0
        else:
            NoiseImg[rX, rY] = 255
    return NoiseImg


img = cv2.imread('lenna.png',0)
img2 = cv2.imread('lenna.png',0)
dst = func(img, 0.3)

cv2.imshow("img", img2)
cv2.imshow("P_noise", dst)

#调用skimage模块
img3 = cv2.imread('lenna.png',0)
noise_pep_img=util.random_noise(img3,mode="s&p",amount = 0.3,salt_vs_pepper = 0.5)
cv2.imshow("noise_pep_img",noise_pep_img)




cv2.waitKey(0)
