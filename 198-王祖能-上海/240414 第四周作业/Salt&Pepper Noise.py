'''
椒盐噪声 salt and pepper noise
'''
import numpy
import cv2
import random

import numpy as np


def SP(src, percentage):
    dst = src.copy()
    h, w = dst.shape
    num = int(h * w * percentage)
    for i in range(num):
        x = random.randint(0, h-1)
        y = random.randint(0, w-1)
        # random.random生成随机浮点数[0,1)，控制概率生成0/255
        if random.random() < 0.5:  # 左闭右开，Return the next random floating point number in the range [0.0, 1.0)
            dst[x, y] = 0
        elif random.random() >= 0.5:
            dst[x, y] = 255
    return dst


img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_sp = SP(img_gray, 0.5)  # percentage可以大于1，相当于对图像进行多轮椒盐加噪处理，=2时还隐约可见，之后逐渐模糊原有图像特征
cv2.imshow('Salt and Pepper Noise 50%', np.hstack([img_gray, img_sp]))
cv2.imwrite('Salt and Pepper Noise 50%.png', np.hstack([img_gray, img_sp]))
cv2.waitKey()

# 彩色图像的椒盐加噪
img = cv2.imread('lenna.png')
b, g, r = cv2.split(img)
bSP, gSP, rSP = SP(b, 2), SP(g, 2), SP(r, 2)
img_sp_color = cv2.merge((bSP, gSP, rSP))
cv2.imshow('Salt and Pepper Noise', np.hstack([bSP, gSP, rSP]))
cv2.waitKey()
cv2.imshow('Salt and Pepper Noise', np.hstack([img, img_sp_color]))
cv2.waitKey()
