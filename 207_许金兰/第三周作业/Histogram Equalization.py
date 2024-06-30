"""
@author: 207-xujinlan
直方图均衡化
"""

import numpy as np
import cv2


def Histogram_eq(img):
    """
    直方图均衡化处理图片
    :param img:BGR图片
    :return: 均衡化后的图片
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片灰度化
    w, h = gray.shape
    size = w * h  # h图片大小
    Ni = np.zeros(256, dtype='int')  # 像素点个数
    new_gray = np.zeros((w, h), dtype='uint8')  # 新图片
    # 1.计算每个像素点的个数
    for i in range(w):
        for j in range(h):
            c = gray[i, j]
            Ni[c] += 1
    sumPi = 0
    Ni_after = np.zeros(256, dtype='int')
    # 2.累加计算，并求映射关系
    for k in range(256):
        if Ni[k] > 0:
            Pi = float(Ni[k]) / size
            sumPi += Pi
            Ni_after[k] = int(sumPi * 256 - 1)
    # 3.新图生成
    for i in range(w):
        for j in range(h):
            c = gray[i, j]
            new_gray[i, j] = Ni_after[c]
    return new_gray


if __name__ == '__main__':
    img = cv2.imread("lenna.png", 1)
    new_img = Histogram_eq(img)
    cv2.imshow('Histogram Equalization', new_img)
    cv2.waitKey(0)
