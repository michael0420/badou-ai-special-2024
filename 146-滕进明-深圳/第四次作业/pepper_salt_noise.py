# 椒盐噪声
import cv2
import numpy as np
import random
from numpy import shape

def pepper_salt_noise(src, percentage):
    ps_img = src
    ps_random_range = int(percentage * ps_img.shape[0] * ps_img.shape[1]) # 确定椒盐噪声的像素数目

    for i in range(ps_random_range):
        # 找到需要被椒盐噪声处理的图像坐标， -1的处理是不处理图像图像边缘
        ps_img_x = random.randint(0, src.shape[0] - 1)
        ps_img_y = random.randint(0, src.shape[1] - 1)

        if random.random() >= 0.5: # 随机浮点数如果大于0.5，就直接将色值置为255, 另一种则是直接将色值置为0， 两种情况概率各为一半
            ps_img[ps_img_x, ps_img_y] = 255
        else:
            ps_img[ps_img_x, ps_img_y] = 0
    return ps_img

img = cv2.imread('lenna.png', 0)
ps_img = pepper_salt_noise(img, 0.2)

img2 = cv2.imread('lenna.png')
gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray_img", gray_img)
cv2.imshow("ps_img", ps_img)
cv2.waitKey(0)