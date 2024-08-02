# -*- coding:utf-8 -*-

__author__ = 'Young'

import cv2
import numpy as np

"""
第三周作业
1.实现双线性插值 
2.实现直方图均衡化 
3.实现sobel边缘检测
"""


class MyImage(object):
    def __init__(self, img_path, flag=1):
        self._img_path = img_path
        self.image = self.__read_img(flag)
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.channels = 1 if len(self.image.shape) == 2 else 3

    def __read_img(self, flag):
        return cv2.imread(self._img_path, flags=flag)

    def convert_to_gray_image(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def zoom(self, shape):
        """
        双线性插值实现图片放大缩小
        :param shape:
        :return:
        """
        self.__check_zoom_param(shape)
        if self.__is_target_image_shape_not_change(shape):
            return self.image.copy()
        else:
            return self.__build_zoom_image(shape)

    def __check_zoom_param(self, shape):
        if shape is None or len(shape) > 3 or shape[0] <= 0 or shape[1] <= 0 or (len(shape) == 3 and shape[2] <= 0):
            raise ValueError('Invalid zoom height and width')

    def __is_target_image_shape_not_change(self, shape):
        return self.image.shape[0] == shape[0] and self.image.shape[1] == shape[1]

    def __build_zoom_image(self, shape):
        target_img_shape = self.__generate_target_image_shape(shape)
        target_img = self.__build_empty_image(target_img_shape, np.uint8)
        self.__fill_image(target_img)
        return target_img

    def __generate_target_image_shape(self, shape):
        return shape[0], shape[1], shape[2] if len(shape) == 3 else self.channels

    def __build_empty_image(self, shape, dtype):
        return np.zeros(shape, dtype)

    def __calculate_image_index(self, index, scale_factor):
        img_x = self.__calculate_index(index[0], scale_factor[0], self.height)
        img_y = self.__calculate_index(index[1], scale_factor[1], self.width)
        return img_x, img_y

    def __calculate_index(self, h_index, scale_v, max_index):
        index = int(h_index * scale_v + 0.5)
        if index < 0:
            index = 0
        elif index > max_index:
            index = max_index
        return index

    def __fill_image(self, image):
        scale_factor = self.__get_scale_factor(image.shape)
        for c in range(image.shape[2]):
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    image[y, x, c] = self.__calculate_pixel_value((y, x, c), scale_factor)

    def __get_scale_factor(self, shape):
        return float(self.height) / shape[0], float(self.width) / shape[1]

    def __calculate_pixel_value(self, shape, scale_factor):
        src_c = self.__get_src_coordinate(shape, scale_factor)
        src_0 = self.__get_1st_coordinate(src_c)
        src_1 = self.__get_2nd_coordinate(src_0)
        temp0 = (src_1[1] - src_c[1]) * self.image[src_0[0], src_0[1], shape[2]] + (src_c[1] - src_0[1]) * self.image[
            src_0[0], src_1[1], shape[2]]
        temp1 = (src_1[1] - src_c[1]) * self.image[src_1[0], src_0[1], shape[2]] + (src_c[1] - src_0[1]) * self.image[
            src_1[0], src_1[1], shape[2]]
        return int((src_1[0] - src_c[0]) * temp0 + (src_c[0] - src_0[0]) * temp1)

    def __get_src_coordinate(self, shape, scale_factor):
        return (shape[0] + 0.5) * scale_factor[0] - 0.5, (shape[1] + 0.5) * scale_factor[1] - 0.5

    def __get_1st_coordinate(self, coordinate):
        return int(np.floor(coordinate[0])), int(np.floor(coordinate[1]))

    def __get_2nd_coordinate(self, coordinate):
        return min(coordinate[0] + 1, self.height - 1), min(coordinate[1] + 1, self.width - 1)

    def equalize_hist(self, equalize_channels=2):
        """
        实现直方图均衡化
        :param equalize_channels:
        :param self:
        :return:
        """
        self.__check_image()
        return self.__equalize_hist(equalize_channels)

    def __check_image(self):
        if self.image is None:
            raise ValueError('Image cannot be null')

    def __equalize_hist(self, equalize_channels):
        if equalize_channels == 2:
            return self.__equalize_gray_image()
        elif equalize_channels == 3:
            return self.__equalize_image()
        else:
            raise ValueError('Not support current channel number: ' + equalize_channels)

    def __equalize_image(self):
        b, g, r = cv2.split(self.image)
        bh = cv2.equalizeHist(b)
        gh = cv2.equalizeHist(g)
        rh = cv2.equalizeHist(r)
        return cv2.merge((bh, gh, rh))

    def __equalize_gray_image(self):
        img = cv2.imread(self._img_path, flags=1)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.equalizeHist(gray_img)

    def detect_image_edge(self, direction='X', alpha=0.5, beta=0.5, gamma=0):
        self.__check_image()
        return self.__detect_edge(direction, alpha, beta, gamma)

    def __detect_edge(self, direction, alpha, beta, gamma):
        if direction == 'X':
            return self.__detect_x_edge()
        elif direction == 'Y':
            return self.__detect_y_edge()
        elif direction == 'All':
            return self.__detect_all_edge(alpha, beta, gamma)
        else:
            raise ValueError('Invalid direction param.')

    def __detect_x_edge(self):
        x = cv2.Sobel(self.image, cv2.CV_16S, 1, 0)
        return cv2.convertScaleAbs(x)

    def __detect_y_edge(self):
        y = cv2.Sobel(self.image, cv2.CV_16S, 0, 1)
        return cv2.convertScaleAbs(y)

    def __detect_all_edge(self, alpha, beta, gamma):
        abs_x = self.__detect_x_edge()
        abs_y = self.__detect_y_edge()
        print(alpha)
        print(beta)
        print(gamma)
        return cv2.addWeighted(abs_x, alpha, abs_y, beta, gamma)
