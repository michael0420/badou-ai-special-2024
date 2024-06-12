import cv2
import numpy as np


def chazhihaxi(img):
    chazhi_img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray_img = cv2.cvtColor(chazhi_img, cv2.COLOR_BGR2GRAY)
    str = ''
    for i in range(8):
        for j in range(8):
            if gray_img[i, j] > gray_img[i, j + 1]:
                str = str + "1"
            else:
                str = str + "0"
    return str


img2 = cv2.imread('lenna.png')
s = chazhihaxi(img2)
print(s)
