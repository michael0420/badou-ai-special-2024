# -*- coding:utf-8 -*-

__author__ = 'Young'

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import ImageFilterUtil as IFUtil


class MyImage(object):
    def __init__(self, path):
        """
        设置这么多参数是因为：
            1. 方便使用，入参不用传来传去
            2. 可以print查看中间结果
        :param path:
        """
        self.__path = path
        self.image = self.__read_image()
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.gray_image = None  # 灰度图
        self.sigma = 0.5  # 高斯滤波参数
        self.dim = 5  # 高斯滤波的参数，滤波核的大小，5表示5*5
        self.filtered_image = None  # 滤波后的图像
        self.filter_kernel = None  # canny中用的滤波器，默认为高斯滤波
        self.image_padding = None  # 填充的边缘
        self.nms_image = None  # 极大值抑制后的图像
        self.sobel_x_image = None  # sobel检测后的图像 x
        self.sobel_y_image = None  # sobel检测后的图像 y
        self.sobel_image = None  # sobel检测后的图像
        self.canny_image = None  # canny检测最终图像
        self.angle = None  # 梯度夹角，计算时不用tan，而是用y/x标识
        self.low_boundary = None  # 双阈值检测的低阈值
        self.high_boundary = None  # 双阈值检测的高阈值

    def __read_image(self):
        return cv2.imread(self.__path)

    def gray(self):
        """
        对于.png图片
        用cv2.cvtColor()方法得到的灰度图，张量的值集合是整数集合，
        plt.imread()得到的图像的张量，是浮点数，乘以255之后，再用mean(axis=-1)均值灰度化得到图像张量的值集合也是浮点数集合
        因此，通过同一套canny detail流程计算得到的边缘检测图，最终会有些不一样；相对来说，浮点数集合的张量的最终效果会更清晰一些
        :return:
        """
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def canny(self, threshold1, threshold2):
        self.gray()
        return cv2.Canny(self.gray, threshold1, threshold2)

    def set_gaussian_param(self, sigma, dim):
        self.sigma = sigma
        self.dim = dim

    def canny_detail(self):
        self.__gray_plt()
        self.__smooth_gray_image()
        self.__detect_edge()
        self.__non_maximum_suppression()
        self.__dualthreshold_detection()

    def __gray_plt(self):
        """
        和老师的代码保持一致，并对比老师代码的中间结果，使用这个方法进行灰度化。
        实际灰度化方式有很多，可以结合实际应用场景选择，也可以探索一下cv2和plt库在读取图片和灰度化图片方面的异同
        :return:
        """
        gray_img = plt.imread(self.__path)
        if self.__path[-4:] == '.png':  # png图片的存储格式是0-1的浮点数，需要扩展到0-255范围在做计算
            gray_img = gray_img * 255
        self.gray_image = gray_img.mean(axis=-1)

    def __smooth_gray_image(self):
        """
        默认采用高斯滤波来平滑图像，平滑图像作用仅为了提升图像质量，方便后续边缘检测，如果图像质量本身很好，这步可以省略
        :return:
        """
        self.__set_filter()
        self.__init_filtered_image()
        self.__padding_gray_image()
        self.__fill_filtered_image()

    def __set_filter(self):
        self.filter_kernel = IFUtil.gaussian_kernel(self.dim, self.sigma)

    def __init_filtered_image(self):
        self.filtered_image = np.zeros(self.gray_image.shape)

    def __padding_gray_image(self):
        pad_radius = self.dim // 2
        self.image_padding = np.pad(self.gray_image, ((pad_radius, pad_radius), (pad_radius, pad_radius)), 'constant')

    def __fill_filtered_image(self):
        for i in range(self.height):
            for j in range(self.width):
                filtered_matrix = self.image_padding[i:i + self.dim, j:j + self.dim] * self.filter_kernel
                self.filtered_image[i, j] = np.sum(filtered_matrix)

    def __detect_edge(self):
        self.__init_sobel_images()
        self.__fill_sobel_images()

    def __init_sobel_images(self):
        self.sobel_x_image = np.zeros(self.filtered_image.shape)
        self.sobel_y_image = np.zeros([self.height, self.width])
        self.sobel_image = np.zeros(self.filtered_image.shape)
        self.image_padding = np.pad(self.filtered_image, ((1, 1), (1, 1)), 'constant')

    def __fill_sobel_images(self):
        for i in range(self.height):
            for j in range(self.width):
                self.sobel_x_image[i, j] = np.sum(self.image_padding[i:i + 3, j:j + 3] * IFUtil.sobel_kernel_x())
                self.sobel_y_image[i, j] = np.sum(self.image_padding[i:i + 3, j:j + 3] * IFUtil.sobel_kernel_y())
                self.sobel_image[i, j] = np.sqrt(self.sobel_x_image[i, j] ** 2 + self.sobel_y_image[i, j] ** 2)

    def __non_maximum_suppression(self):
        self.__init_nms_image()
        self.__set_angle()
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                is_edge = True  # 默认当前点为边缘点
                temp = self.sobel_image[i - 1:i + 2, j - 1:j + 2]  # 设置当前像素点的8个领域像素
                if self.angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                    num_1 = (temp[0, 1] - temp[0, 0]) / self.angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 1] - temp[2, 2]) / self.angle[i, j] + temp[2, 1]
                    if not (self.sobel_image[i, j] > num_1 and self.sobel_image[i, j] > num_2):
                        is_edge = False
                elif self.angle[i, j] >= 1:
                    num_1 = (temp[0, 2] - temp[0, 1]) / self.angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 0] - temp[2, 1]) / self.angle[i, j] + temp[2, 1]
                    if not (self.sobel_image[i, j] > num_1 and self.sobel_image[i, j] > num_2):
                        is_edge = False
                elif self.angle[i, j] > 0:
                    num_1 = (temp[0, 2] - temp[1, 2]) * self.angle[i, j] + temp[1, 2]
                    num_2 = (temp[2, 0] - temp[1, 0]) * self.angle[i, j] + temp[1, 0]
                    if not (self.sobel_image[i, j] > num_1 and self.sobel_image[i, j] > num_2):
                        is_edge = False
                elif self.angle[i, j] < 0:
                    num_1 = (temp[1, 0] - temp[0, 0]) * self.angle[i, j] + temp[1, 0]
                    num_2 = (temp[1, 2] - temp[2, 2]) * self.angle[i, j] + temp[1, 2]
                    if not (self.sobel_image[i, j] > num_1 and self.sobel_image[i, j] > num_2):
                        is_edge = False
                if is_edge:
                    self.nms_image[i, j] = self.sobel_image[i, j]

    def __init_nms_image(self):
        self.nms_image = np.zeros(self.sobel_image.shape)

    def __set_angle(self):
        self.sobel_x_image[self.sobel_x_image == 0] = 0.00000001
        self.angle = self.sobel_y_image / self.sobel_x_image

    def __is_not_edge_1(self, i, j, pixel_nhood):
        num_1 = (pixel_nhood[0, 1] - pixel_nhood[0, 0]) / self.angle[i, j] + pixel_nhood[0, 1]
        num_2 = (pixel_nhood[2, 1] - pixel_nhood[2, 2]) / self.angle[i, j] + pixel_nhood[2, 1]
        return not (self.sobel_image[i, j] > num_1 and self.sobel_image[i, j] > num_2)

    def __is_not_edge_2(self, i, j, pixel_nhood):
        num_1 = (pixel_nhood[0, 2] - pixel_nhood[0, 1]) / self.angle[i, j] + pixel_nhood[0, 1]
        num_2 = (pixel_nhood[2, 0] - pixel_nhood[2, 1]) / self.angle[i, j] + pixel_nhood[2, 1]
        return not (self.sobel_image[i, j] > num_1 and self.sobel_image[i, j] > num_2)

    def __is_not_edge_3(self, i, j, pixel_nhood):
        num_1 = (pixel_nhood[0, 2] - pixel_nhood[1, 2]) / self.angle[i, j] + pixel_nhood[1, 2]
        num_2 = (pixel_nhood[2, 0] - pixel_nhood[1, 0]) / self.angle[i, j] + pixel_nhood[1, 0]
        return not (self.sobel_image[i, j] > num_1 and self.sobel_image[i, j] > num_2)

    def __is_not_edge_4(self, i, j, pixel_nhood):
        num_1 = (pixel_nhood[1, 0] - pixel_nhood[0, 0]) / self.angle[i, j] + pixel_nhood[1, 0]
        num_2 = (pixel_nhood[1, 2] - pixel_nhood[2, 2]) / self.angle[i, j] + pixel_nhood[1, 2]
        return not (self.sobel_image[i, j] > num_1 and self.sobel_image[i, j] > num_2)

    def __dualthreshold_detection(self):
        self.canny_image = np.copy(self.nms_image)
        self.low_boundary = self.sobel_image.mean() * 0.5
        self.high_boundary = self.low_boundary * 3
        zhan = []
        for i in range(1, self.canny_image.shape[0] - 1):  #  不考虑外圈
            for j in range(1, self.canny_image.shape[1] - 1):
                if self.canny_image[i, j] >= self.high_boundary:
                    self.canny_image[i, j] = 255
                    zhan.append((i, j))
                elif self.canny_image[i, j] <= self.low_boundary:
                    self.canny_image[i, j] = 0
        while not len(zhan) == 0:
            temp_1, temp_2 = zhan.pop()
            a = self.canny_image[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
            if (a[0, 0] < self.high_boundary) and (a[0, 0] > self.low_boundary):
                self.canny_image[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
                zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
            if (a[0, 1] < self.high_boundary) and (a[0, 1] > self.low_boundary):
                self.canny_image[temp_1 - 1, temp_2] = 255
                zhan.append([temp_1 - 1, temp_2])
            if (a[0, 2] < self.high_boundary) and (a[0, 2] > self.low_boundary):
                self.canny_image[temp_1 - 1, temp_2 + 1] = 255
                zhan.append([temp_1 - 1, temp_2 + 1])
            if (a[1, 0] < self.high_boundary) and (a[1, 0] > self.low_boundary):
                self.canny_image[temp_1, temp_2 - 1] = 255
                zhan.append([temp_1, temp_2 - 1])
            if (a[1, 2] < self.high_boundary) and (a[1, 2] > self.low_boundary):
                self.canny_image[temp_1, temp_2 + 1] = 255
                zhan.append([temp_1, temp_2 + 1])
            if (a[2, 0] < self.high_boundary) and (a[2, 0] > self.low_boundary):
                self.canny_image[temp_1 + 1, temp_2 - 1] = 255
                zhan.append([temp_1 + 1, temp_2 - 1])
            if (a[2, 1] < self.high_boundary) and (a[2, 1] > self.low_boundary):
                self.canny_image[temp_1 + 1, temp_2] = 255
                zhan.append([temp_1 + 1, temp_2])
            if (a[2, 2] < self.high_boundary) and (a[2, 2] > self.low_boundary):
                self.canny_image[temp_1 + 1, temp_2 + 1] = 255
                zhan.append([temp_1 + 1, temp_2 + 1])
        for i in range(self.canny_image.shape[0]):
            for j in range(self.canny_image.shape[1]):
                if self.canny_image[i, j] != 0 and self.canny_image[i, j] != 255:
                    self.canny_image[i, j] = 0
