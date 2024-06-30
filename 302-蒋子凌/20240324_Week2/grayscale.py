# -*- coding: utf-8 -*-
"""
HOMEWORK

School:BadouAI
Student:302-蒋子凌
Week2
Homework 1/3: grayscale
"""

import numpy as np
# import matplotlib.pyplot as plt
# import cv2


def grayscale_brg(img):
    '''
    Function:
        1.grayscale the BRG img
        2.return the gray img
    Gray = R0.3+G0.59+B0.11
    '''
    
    h,w = img.shape[:2]  # get the height and width of the img
    
    img_gray = np.zeros([h,w], dtype=img.dtype)  # create an empty img based on the shape of the given img
    
    for i in range(h):  # iterate the height
    
        for j in range(w):  # iterate the width
        
            pixel = img[i,j]
            # print(pixel)
            
            # img_gray[i,j] = pixel[0]*0.3 + pixel[1]*0.59 + pixel[2]*0.11   # 0.3R 0.59G 0.11B, for img read in by plt
            img_gray[i,j] = pixel[0]*0.11 + pixel[1]*0.59 + pixel[2]*0.3   # 0.11B 0.59G 0.3R, for img read in by opencv
            
    return img_gray


def grayscale_rgb(img):
    '''
    grayscale the RGB img and return the gray img
    Gray = R0.3+G0.59+B0.11
    '''
    
    h,w = img.shape[:2]  # get the height and width of the img
    
    img_gray = np.zeros([h,w], dtype=img.dtype)  # create an empty img bosed on the shape of the given img
    
    for i in range(h): # iterate the height
    
        for j in range(w):  # iterate the width
        
            pixel = img[i,j]
            # print(pixel)
            
            img_gray[i,j] = pixel[0]*0.3 + pixel[1]*0.59 + pixel[2]*0.11   # 0.3R 0.59G 0.11B, for img read in by plt
            # img_gray[i,j] = pixel[0]*0.11 + pixel[1]*0.59 + pixel[2]*0.3   # 0.11B 0.59G 0.3R, for img read in by opencv
            
    return img_gray