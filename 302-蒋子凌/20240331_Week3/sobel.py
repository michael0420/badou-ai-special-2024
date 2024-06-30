# -*- coding: utf-8 -*-
"""
HOMEWORK

School:BadouAI
Student:302-蒋子凌
Week3
Homework 3/3: sobel edge detection
"""

# import numpy as np
# import matplotlib.pyplot as plt
import cv2


def get_edge_with_sobel(img, alpha=0.5, beta=0.5):
    '''
    get edge of img with sobel
    '''
    
    # do sobel on both direction of img, using cv2.CV_16S data type for data out of bound [0,255]
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    
    # convert back to uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    
    # dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
    dst = cv2.addWeighted(absX, alpha, absY, beta, 0)  # form weighted 2 direction of img into one
    
    return dst

if __name__ == '__main__':
    
    img = cv2.imread("lenna.png", 0)  # read in img
    sobel_img = get_edge_with_sobel(img)
    cv2.imwrite('lenna_sobel.jpg', sobel_img)