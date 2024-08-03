# -*- coding:utf-8 -*-

__author__ = 'Young'

import cv2
from matplotlib import pyplot as plt
import numpy as np


def equalize_hist(img):
    """
    equalize the histogram of image.
    :param img:
    :return: hist
    """
    __check_image(img)
    return __equalize_hist(img)


def __check_image(img):
    if img is None:
        raise ValueError('Image cannot be null')
    if img.shape is None:
        raise ValueError('Invalid image object')


def __equalize_hist(img):
    if img.shape[2] == 3:
        return __equalize_image(img)
    else:
        return __equalize_gray_image(img)


def __equalize_image(img):
    b, g, r = cv2.split(img)
    bh = cv2.equalizeHist(b)
    gh = cv2.equalizeHist(g)
    rh = cv2.equalizeHist(r)
    return cv2.merge((bh, gh, rh))


def __equalize_gray_image(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def compare_equalized_image(img_arr):
    """
    Compare the original image and the equalized image.
    :param img_arr: the image array
    :return: None
    """
    cv2.imshow('Comparing Equalization', np.hstack(img_arr))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_equalization_histogram(hist):
    """
    Show the pixel distribution histogram of equalized image
    :param hist:
    :return:
    """
    plt.figure()
    plt.hist(hist.ravel(), 256)
    plt.show()

