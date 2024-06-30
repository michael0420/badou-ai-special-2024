# -*- coding: utf-8 -*-
"""
HOMEWORK

School:BadouAI
Student:302-蒋子凌
Week2
Homework 2/3: binarization
"""

import numpy as np
# import matplotlib.pyplot as plt
# import cv2


def binarization1(img_gray, threshold_value=0.5):
    '''
    binarization of grayscaled img
    '''
    
    h, w = img_gray.shape  # no need to slice, cause the shape of gray image contains only 2 dimention of height and width
    
    for i in range(h):  # iterate the height
    
        for j in range(w):  # iterate the width
        
            if (img_gray[i, j] <= threshold_value):  # binarization by the threshold value
            
                img_gray[i, j] = 0
                
            else:
                
                img_gray[i, j] = 1
                
    return img_gray


def binarazation2(img_gray, threshold_value=0.5):
    """
    binarization of grayscaled img
    """
    
    img_binary = np.where(img_gray >= threshold_value, 1, 0)
    
    return img_binary