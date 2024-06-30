# -*- coding: utf-8 -*-
"""
HOMEWORK

School:BadouAI
Student:302-蒋子凌
Week3
Homework 1/3: bilinear interpolation
"""

import numpy as np
# import matplotlib.pyplot as plt
import cv2


def bilinear_interpolation(img, target_h, target_w):
    """
    do bilinear interpolation on img
    return img in shape of target_h * target_w
    """
    source_h, source_w, channel = img.shape  # get shape info of img
    # print('source: ' + source_h + 'x' + source_w) 
    # print('target: ' + target_h + 'x' + target_w)
    if (source_h == target_h) and (source_w == target_w):  # return a copy when no need to change
        return img.copy()
    
    target_image = np.zeros((target_h, target_w, 3))  # create an empty container
    scale_x, scale_y = source_w/target_w, source_h/target_h  # calculate the scale ratio
    
    for i in range(channel):  # iterate the channels
        for target_y in range(target_h):  # iterate the height
            for target_x in range(target_w):  # iterate the width
                
                # find the x and y coordinates of source, based on x and y of target image
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                source_x = (target_x + 0.5) * scale_x - 0.5
                source_y = (target_y + 0.5) * scale_y - 0.5
                
                # find the coordinates of the points which will be used to compute the interpolation
                source_x0 = int(np.floor(source_x))
                source_x1 = min(source_x0 + 1, source_w - 1)
                source_y0 = int(np.floor(source_y))
                source_y1 = min(source_y0 + 1, source_h - 1)
 
                # calculate the interpolation
                temp0 = ((source_x1 - source_x) * img[source_y0, source_x0, i] + 
                    (source_x - source_x0) * img[source_y0, source_x1, i])
                temp1 = ((source_x1 - source_x) * img[source_y1, source_x0, i] + 
                    (source_x - source_x0) * img[source_y1, source_x1, i])
                target_image[target_y, target_x, i] = int((source_y1 - source_y) * temp0 + (source_y - source_y0) * temp1)
                
    return target_image
        
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    target = bilinear_interpolation(img,700,700)
    print('target generated')
    # cv2.imshow('bilinear interp',target)
    # cv2.waitKey()
    cv2.imwrite('lenna_bilinear_interpolated.jpg',target)