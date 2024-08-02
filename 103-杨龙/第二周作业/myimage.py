# -*- coding:utf-8 -*-

__author__ = 'Young'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

"""
第二周作业：
1.实现灰度化和二值化 
2.实现最临近插值
"""


class MyImage(object):
    def __init__(self, img_path):
        self._img_path = img_path
        self.image = self.__read_img()
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.channels = self.image.shape[2]

    def gray_image(self, convert_way=''):
        """
        实现灰度化
        :param self:
        :param convert_way:灰度化方式(cv2, pyplot, 默认为用色测比例实现)
        :return:
        """
        if convert_way == 'cv2':
            return self.__build_gray_image_cv2()
        elif convert_way == 'pyplot':
            return self.__build_gray_image_pyplot()
        else:
            return self.__build_gray_image_default()

    def __build_gray_image_cv2(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def __build_gray_image_pyplot(self):
        img = plt.imread(self._img_path)
        return rgb2gray(img)

    def __build_gray_image_default(self):
        gray_img = self.__build_empty_image(list(self.image.shape[:2]), self.image.dtype)
        self.__fill_gray_image(gray_img)
        return gray_img

    def __read_img(self):
        """
        TBD: 由BRG转成RGB之后，imshow()会发现图片变色了。
            这会影响临近插值缩放图片的计算；但对灰度化的视觉结果影响不大。
            这里可以深挖一下为什么？
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        """
        return cv2.imread(self._img_path)

    def __build_empty_image(self, shape, dtype):
        return np.zeros(shape, dtype)

    def __fill_gray_image(self, gray_img):
        height, width = self.image.shape[:2]
        for i in range(height):
            for j in range(width):
                gray_img[i, j] = self.__calculate_grayscale_pixel(self.image[i, j])

    def __calculate_grayscale_pixel(self, rgb):
        return int(rgb[0] * 0.3 + rgb[1] * 0.59 + rgb[2] * 0.11)

    def binarize(self):
        """
        二值化图片
        :param self:
        :return:
        """
        gray_img = self.gray_image('pyplot')
        self.__binarize(gray_img)
        return np.where(gray_img >= 0.5, 1, 0)

    def __binarize(self, gray_image):
        for i in range(gray_image.shape[0]):
            for j in range(gray_image.shape[1]):
                gray_image[i, j] = 0 if gray_image[i, j] <= 0.5 else 1

    def zoom(self, shape):
        """
        临近插值实现图片放大缩小
        :param self:
        :param shape:
        :return:
        """
        self.__check_zoom_param(shape)
        if self.__is_image_shape_not_change(shape):
            return self.image.copy()
        else:
            return self.__build_zoom_image(shape)

    def __check_zoom_param(self, shape):
        if shape is None or len(shape) > 3 or shape[0] <= 0 or shape[1] <= 0 or (len(shape) == 3 and shape[2] <= 0):
            raise ValueError('Invalid zoom height and width')

    def __is_image_shape_not_change(self, shape):
        return self.image.shape[0] == shape[0] and self.image.shape[1] == shape[1]

    def __build_zoom_image(self, shape):
        target_img_shape = self.__generate_target_image_shape(shape)
        target_img = self.__build_empty_image(target_img_shape, np.uint8)
        self.__fill_image(target_img)
        return target_img

    def __generate_target_image_shape(self, shape):
        return shape[0], shape[1], shape[2] if len(shape) == 3 else self.channels

    def __fill_image(self, tar_img):
        scale_factor = self.__get_scale_factor(tar_img.shape)
        for i in range(tar_img.shape[0]):
            for j in range(tar_img.shape[1]):
                img_h, img_w = self.__calculate_image_index((i, j), scale_factor)
                tar_img[i, j] = self.image[img_h, img_w]

    def __get_scale_factor(self, shape):
        return shape[0] / self.height, shape[1] / self.width

    def __calculate_image_index(self, index, scale_factor):
        img_h = self.__calculate_index(index[0], scale_factor[0], self.height)
        img_w = self.__calculate_index(index[1], scale_factor[1], self.width)
        return img_h, img_w

    def __calculate_index(self, index_v, scale_v, max_index):
        index = int(index_v / scale_v + 0.5)
        if index < 0:
            index = 0
        elif index > max_index:
            index = max_index
        return index
