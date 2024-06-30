# -*- coding: utf-8 -*-
"""
HOMEWORK

School:BadouAI
Student:302-蒋子凌
Week2
Homework
main program
"""

# import numpy as np
import matplotlib.pyplot as plt
# import cv2

from grayscale import grayscale_rgb
from binarization import binarazation2
from nearest_neighbour_interpolation import cal_nni


if __name__ == '__main__':
    
    # Done with plt
    
    plt.subplot(221)  # the 1st plot in grill 2*2
    img = plt.imread('lenna.png')  # read in the source image
    plt.imshow(img)  # show the plot
    
    plt.subplot(222)  # the 2nd plot in grill 2*2
    img_gray = grayscale_rgb(img)  # RGB image cause it's read in by plt
    plt.imshow(img_gray, cmap='gray')  # set cmap='gray' when showing gray image with plt 
    
    plt.subplot(223)  # the 3rd plot in grill 2*2
    img_binary = binarazation2(img_gray)  # grayscale it
    plt.imshow(img_binary, cmap='gray')  # set cmap='gray' when showing gray image with plt
    
    plt.subplot(224)  # the 4th plot in grill 2*2
    img_nni = cal_nni(img)
    plt.imshow(img_nni)
    
    plt.savefig('lanna_result.jpg')
    plt.close()