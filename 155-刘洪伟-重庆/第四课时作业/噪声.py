# _*_ coding: UTF-8 _*_
# @Time: 2024/4/17 15:56
# @Author: iris
import cv2
from skimage import util

if __name__ == '__main__':
    image = cv2.imread('../data/lenna.png')
    dst = util.random_noise(image, mode='s&p')
    img = cv2.imread('../data/lenna.png')
    cv2.imshow('source', image)
    cv2.imshow('noice', dst)
    cv2.waitKey(0)
