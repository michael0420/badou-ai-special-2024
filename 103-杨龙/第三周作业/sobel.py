#-*-coding:utf-8-*-

import cv2
import numpy as np

def sobel_image(img):
    absx_img = __calculate_scale_abs(img,1,0)
    absy_img = __calculate_scale_abs(img,0,1)
    dst_img = cv2.addWeighted(absx_img, 0.5,absy_img,0.5,0)
    return absx_img,absy_img,dst_img


def __calculate_scale_abs(img,dx,dy):
    gradient =__get_gradient(img,dx,dy)
    return cv2.convertScaleAbs(gradient)

def __get_gradient(img,dx,dy):
    return cv2.Sobel(img, cv2.CV_16S, dx, dy)


if __name__ =='__main__':
    img = cv2.imread("lenna.png", 0)
    soble_result = sobel_image(img)
    cv2.imshow('origial img',img)
    cv2.imshow('absx',soble_result[0])
    cv2.imshow('absy',soble_result[1])
    cv2.imshow('result',soble_result[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

