import numpy as np
import cv2
from numpy import shape
import random

def GaussianNoise(src,sigma,mean,percetage):
    # 获取需要需要循环的像素数量
    per_pixel = int(percetage * src.shape[0] * src.shape[1])
    # 遍历像素数量
    for i in range(per_pixel):
        # 获取横、纵坐标，-1是因为坐标是从0开始
        x = random.randint(0, src.shape[0]-1)
        y = random.randint(0, src.shape[1]-1)
        # 图片元素值加上高斯随机数
        src[x, y] = src[x, y] + random.gauss(sigma, mean)
        # 像素值边界重置，因为像素值范围是0-255
        if src[x, y] > 255:
            src[x, y] = 255
        elif src[x, y] < 0:
            src[x, y] = 0

    return src

# 获取lenna的灰度图片
img = cv2.imread("lenna.png",0)
print(img)
gnoise_img = GaussianNoise(img,2,4,0.6)
img3 = cv2.imread('lenna.png')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

cv2.imshow("gnoise",gnoise_img)
cv2.imshow('img3',img3)
cv2.waitKey(0)
