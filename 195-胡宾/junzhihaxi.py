import cv2
import numpy as np


def junzhihaxi(img):
    # 先将图片缩小
    resize_img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 灰度化
    gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    # 求和
    s = 0
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            s = s + gray_img[i, j]
    avg = s/64
    str = ''
    for i in range(8):
        for j in range(8):
            if gray_img[i, j] > avg:
                str = str + '1'
            else:
                str = str + '0'

    return str


img1=cv2.imread('lenna.png')
s = junzhihaxi(img1)
print(s)

