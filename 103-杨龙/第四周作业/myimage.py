# -*- coding:utf-8 -*-

__author__ = "Young"

import random

import cv2


class MyImage(object):
    def __init__(self, path):
        self.__path = path
        self.image = self.__read_image()
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def __read_image(self):
        return cv2.imread(self.__path)

    def noise_image(self, percentage, mode='gaussian', **kw):
        """
        :param percentage:
        :param mode: gaussian, salt, pepper, salt&pepper
        :param kw:
        :return:
        """
        image = self.gray_image()
        noise_count = self.__get_noise_number(percentage)
        noise_func = self.__get_noise_func(mode)
        self.__fill_image_noise(image, noise_count, noise_func, **kw)
        return image

    def gray_image(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def __get_noise_number(self, percentage):
        return int(percentage * self.height * self.width)

    def __get_noise_func(self, mode):
        """
        :param mode: gaussian, salt, pepper, salt&pepper
        :return:
        """
        if mode == 'salt':
            return self.__get_salt_noise
        elif mode == 'pepper':
            return self.__get_pepper_noise
        elif mode == "salt&pepper":
            return self.__get_salt_pepper_noise
        elif mode == "gaussian":
            return self.__calculate_gaussian_noise
        else:
            raise ValueError('Invalid model:%s' % mode)

    def __get_salt_noise(self, *param, **kw):
        return 255

    def __get_pepper_noise(self, *param, **kw):
        return 0

    def __get_salt_pepper_noise(self, *param, **kw):
        if random.random() <= 0.5:
            return self.__get_pepper_noise()
        else:
            return self.__get_salt_noise()

    def __calculate_gaussian_noise(self, pixel, means=None, sigma=None):
        noise = pixel + random.gauss(means, sigma)
        return self.__limit_noise_range(noise)

    def __limit_noise_range(self, noise):
        if noise < 0:
            return 0
        elif noise > 255:
            return 255
        else:
            return noise

    def __fill_image_noise(self, image, count, noise_func, **kw):
        for i in range(count):
            x, y = self.__get_random_coordinates()
            image[x, y] = noise_func(pixel=image[x, y], **kw)

    def __get_random_coordinates(self):
        return random.randint(0, self.height - 1), random.randint(0, self.width - 1)  # 高斯噪声一般不处理边缘值
