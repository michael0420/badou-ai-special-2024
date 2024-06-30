# -*- coding: utf-8 -*-
"""
HOMEWORK

School:BadouAI
Student:302-蒋子凌
Week2
Homework 3/3: nearest_neighbour_interpolation
"""

import numpy as np
# import matplotlib.pyplot as plt
# import cv2

def cal_nni(img, targetH=800, targetW=800):
    '''
    change the resolution of a image with NNI(Nearest Neighbor Interpolation)
    '''
    
    h, w, channels = img.shape  # obtain the height, width and channels information
    
    # emptyImg = np.zeros([targetH,targetW,channels], dtype=np.uint8)  # create an empty container
    emptyImg = np.zeros([targetH,targetW,channels])  # create an empty container
    
    scaleH = h/targetH  # calculate the scale ratio of height direction
    scaleW = w/targetW  # calculate the scale ratio of width direction
    # print(scaleH, scaleW)  # debug
    
    for i in range(targetH):  # iterate on height
        
        for j in range(targetW):  # iterate on width
            
            x = int(i*scaleH + 0.5)  # calculate the positon of source pixel based on the scale ratio
            y = int(j*scaleW + 0.5)  # + 0.5 for reducing floor error, which is similar to round a number with int() function
            # print(x,y)  # debug
            
            # print(img[x,y])  # debug
            # print(i,j,emptyImg[i,j])  # debug
            emptyImg[i,j] = img[x,y]  # give the source pixel to its corresponding target pixel
            # print(emptyImg[i,j])  # debug
            # print(i,j,'done')  # debug
            
    return emptyImg