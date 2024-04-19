import cv2
import numpy as np
from matplotlib import pyplot as plt

# 彩色图像直方图均衡化
def fun(img):
    (b,g,r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)

    dst_img = cv2.merge((bH,gH,rH))

    return dst_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png", 1)
    dst_img = fun(img)
    cv2.imshow("dst_img",dst_img)
    cv2.waitKey(0)