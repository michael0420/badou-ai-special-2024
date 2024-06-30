# -*- coding: utf-8 -*-
"""
HOMEWORK

School:BadouAI
Student:302-蒋子凌
Week3
Homework 2/3: histogram equalization
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def histogram_equalization(img):
    """
    do histogram equalization on all channels with cv2.split + cs2.equalizeHist
    """
    
    (b, g, r) = cv2.split(img)  # split img channels
    bH = cv2.equalizeHist(b)  # do histogram equalization on blue channel
    gH = cv2.equalizeHist(g)  # do histogram equalization on green channel
    rH = cv2.equalizeHist(r)  # do histogram equalization on red channel
    
    equalized_img = cv2.merge((bH, gH, rH))  # merge all channels
    
    return equalized_img  # return equalized img


def histogram_equalization_1(img):
    """
    do histogram equalization on grayscaled img
    """
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale img with cv2.cvtColor
    dst = cv2.equalizeHist(gray)  # equalize the hist of gray img
    
    # hist = cv2.calcHist([dst],[0],None,[256],[0,256])
    # plt.figure()
    # plt.hist(dst.ravel(), 256)
    
    equalized_img = dst  # return equalized img
    # equalized_img = np.hstack([gray, dst])
    
    return equalized_img  # return equalized img


if __name__ == '__main__':  # run if this script file is called from console
    
    img = cv2.imread('lenna.png', 1)  # read in img
    
    equalized_img = histogram_equalization(img)  # do histogram equalization with function
    equalized_img_1 = histogram_equalization_1(img)  # do histogram equalization with function_1
    
    # cv2.imshow('dst_img', equalized_img)  # show img
    # cv2.waitKey(2)  # show img
    
    cv2.imwrite('lenna_equalized.jpg', equalized_img)  # output result img of func
    cv2.imwrite('lenna_equalized_1.jpg', equalized_img_1)  # output result img of func 1
