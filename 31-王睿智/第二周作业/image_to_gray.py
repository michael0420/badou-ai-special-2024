# -*- coding: utf-8 -*-
"""
作业：1.实现灰度化和二值化
"""

import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# 图片路径
img_path = "lenna.png"


def image_to_gray_by_cv2():
    """
    使用cv2 自带函数实现灰度化
    :return:
    """
    # 加载图片
    img = cv2.imread(img_path)
    # 图片灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 显示图片
    cv2.imshow("cv2 image to gray", img_gray)
    cv2.waitKeyEx(0)


def image_to_gray_by_formula():
    """
    使用RGB公式实现，Gray = R0.3+G0.59+B0.11
    :return:
    """
    # 加载图片
    img = cv2.imread("lenna.png")
    # 获取图片高、款
    h, w = img.shape[:2]
    # 生成与原图片大小一致的且全是0的数组
    img_gray = np.zeros((h, w), img.dtype)
    # 遍历每一个节点
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            # print(m) ->[125 137 226],每个[i,j]点都有RGB的实际值，
            # 按照公式进行换算并将最新的值赋值给空数组img_gray
            # 注：cv2 是BGR格式
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)

    cv2.imshow("custom image gray", img_gray)
    cv2.waitKeyEx(0)


def image_to_gray_by_plt():
    """
    使用matplotlib、skimage实现灰度化
    :return:
    """
    # 加载图片
    img = plt.imread("lenna.png")
    # 将图片转为灰度图
    img_gray = rgb2gray(img)
    # cmap：用于指定渐变色，若是不写则与期望效果不一致
    plt.imshow(img_gray, cmap="gray")
    # 显示效果
    plt.show()


def image_to_binary_cv2():
    """
    通过将图片转为灰度图片，在进行图片二值化处理
    使用cv2自带灰度方式进行二值化，需要注意赋值应为0或255
    :return:
    """
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape[:2]
    img_binary = np.zeros((h, w), img_gray.dtype)
    for i in range(h):
        for j in range(w):
            # 取值范围0-255
            m = img_gray[i, j]
            # 方案一：m<=128,方案二：m/255 < 0.5
            if m <= 128:
                img_binary[i, j] = 0
            else:
                img_binary[i, j] = 255
    cv2.imshow("binary img", img_binary)
    cv2.waitKeyEx(0)


def image_to_binary_skimage():
    """
    通过将图片转为灰度图片，在进行图片二值化处理
    使用rgb2gray实现灰度化，需要注意赋值应为0或1
    :return:
    """
    img = cv2.imread(img_path)
    img_gray = rgb2gray(img)
    h, w = img_gray.shape[:2]
    img_binary = np.zeros((h, w), img_gray.dtype)
    for i in range(h):
        for j in range(w):
            # 取值范围0-1
            m = img_gray[i, j]
            if m <= 0.5:
                img_binary[i, j] = 0
            else:
                img_binary[i, j] = 1
    cv2.imshow("binary img", img_binary)
    cv2.waitKeyEx(0)


def image_to_binary_by_plt():
    """
    使用matpoltlib、numpy实现图片二值化
    :return:
    """
    # 加载图片
    img = plt.imread(img_path)
    # 图片灰度化
    img_gray = rgb2gray(img)
    # 图片二值化
    img_binary = np.where(img_gray >= 0.5, 1, 0)
    plt.imshow(img_binary, cmap="gray")
    plt.show()


if __name__ == '__main__':
    # 图片灰度化
    image_to_gray_by_cv2()
    image_to_gray_by_formula()
    image_to_gray_by_plt()
    # 图片二值化
    image_to_binary_cv2()
    image_to_binary_skimage()
    image_to_binary_by_plt()
