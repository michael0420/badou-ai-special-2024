# -*- coding: utf-8 -*-
"""
@author: zhjd

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray


def turn_to_gray(img):
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)

    for i in range(h):
        for j in range(w):
            ori = img[i, j]
            img_gray[i, j] = int(ori[0] * 0.11 + ori[1] * 0.59 + ori[2] * 0.3)

    return img_gray


def turn_to_binary(img_gray):
    h, w = img_gray.shape[:2]
    img_binary = np.zeros([h, w], img_gray.dtype)

    for i in range(h):
        for j in range(w):
            o = img_gray[i, j]/255
            if o > 0.5:
                img_binary[i, j] = 1
            else:
                img_binary[i, j] = 0

    return img_binary


def turn_to_gray2(img):
    return rgb2gray(img)


def turn_to_binary2(img_gray):
    return np.where(img_gray >= 0.5, 1, 0)


if __name__ == '__main__':
    origin = plt.imread("alex.jpg")
    gray = turn_to_gray(origin)
    binary = turn_to_binary(gray)
    gray2 = turn_to_gray2(origin)
    binary2 = turn_to_binary2(gray2)
    plt.subplot(231)
    # plt.imshow(plt.imread("alex.jpg"))
    plt.imshow(origin)
    plt.subplot(232)
    plt.imshow(gray, cmap='gray')
    plt.subplot(233)
    plt.imshow(binary, cmap='gray')
    plt.subplot(234)
    plt.imshow(gray2, cmap='gray')
    plt.subplot(235)
    plt.imshow(binary2, cmap='gray')
    plt.show()
