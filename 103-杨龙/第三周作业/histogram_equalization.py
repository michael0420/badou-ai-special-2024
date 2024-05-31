#-*-coding:utf-8-*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_gray_image_equalized_histogram(img_path = 'lenna.png'):
    gray_img =__get_gray_img(img_path)
    equalized_img = __equalize_channel_histogram(gray_img)
    dest_hist = __get_img_histogram(equalized_img)
    orig_hist = __get_img_histogram(gray_img)
    __show_equalized_histogram(orig_hist, dest_hist)
    __show_img(gray_img, equalized_img)

def __get_gray_img(img_path):
    img = cv2.imread(img_path, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def __equalize_channel_histogram(img):
    return cv2.equalizeHist(img)

def __get_img_histogram(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])

def __show_equalized_histogram(orig_hist, equalized_hist ):
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(orig_hist)
    plt.title('Original histogram')
    plt.xlim([0,256])

    plt.subplot(1,2,2)
    plt.plot(equalized_hist)
    plt.title('Destination histogram')
    plt.xlim([0,256])

    plt.show()

def __show_img(orig_img, qualized_img):
    cv2.imshow('Image histogram equalization comparing result', np.hstack([orig_img, qualized_img]))
    cv2.waitKey(0)

def show_equalized_image(img_path = 'lenna.png'):
    ori_img = __get_image(img_path)
    equalized_img = __equalize_image_historam(ori_img)
    __show_img(ori_img,equalized_img)

def __get_image(img_path):
    return cv2.imread(img_path,1)

def __equalize_image_historam(img):
    (b,g,r) = cv2.split(img)
    bH = __equalize_channel_histogram(b)
    gH = __equalize_channel_histogram(g)
    rH = __equalize_channel_histogram(r)
    return cv2.merge((bH,gH,rH))

if __name__ =='__main__':
    #show_gray_image_equalized_histogram()
    show_equalized_image()