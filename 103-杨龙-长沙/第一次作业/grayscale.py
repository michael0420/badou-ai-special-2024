# -*- conding: utf8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def convert_to_gray_image(img_path):
    img = cv2.imread(img_path)
    return generate_gray_image(img)


def generate_gray_image(img):
    img_gray = generate_default_gray_image(img)
    init_gray_image(img, img_gray)
    return img_gray


def generate_default_gray_image(img):
    height, width = img.shape[:2]
    return np.zeros([height, width], img.dtype)


def init_gray_image(img, img_gray):
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            rgb = img[i, j]
            img_gray[i, j] = int(rgb[0] * 0.11 + rgb[1] * 0.59 + rgb[2] * 0.3)


def convert_to_gray_image_pyplot(img_path):
    img = plt.imread(img_path)
    img_gray = rgb2gray(img)
    return img_gray


def convert_to_gray_image_cv2(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray
