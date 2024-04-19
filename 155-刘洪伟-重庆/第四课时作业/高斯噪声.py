# _*_ coding: UTF-8 _*_
# @Time: 2024/4/17 15:35
# @Author: iris
import random

import cv2


def gaussian_noise(src, means, sigma, ratio):
    """
    高斯噪声
    :param src: 灰度图
    :param means: 均值
    :param sigma:
    :param ratio: 噪声比例
    :return:
    """
    h, w = src.shape[0], src.shape[1]
    dst_image = src
    scale = int(ratio * h * w)
    for i in range(scale):
        src_x = random.randint(0, h - 1)
        src_y = random.randint(0, w - 1)
        dst_image[src_x, src_y] = dst_image[src_x, src_y] + random.gauss(means, sigma)
        if dst_image[src_x, src_y] > 255:
            dst_image[src_x, src_y] = 255
        elif dst_image[src_x, src_y] < 0:
            dst_image[src_x, src_y] = 0

    return dst_image


if __name__ == '__main__':
    image = cv2.imread('../data/lenna.png', 0)
    dst = gaussian_noise(image, 2, 4, 0.6)
    image = cv2.imread('../data/lenna.png', 0)
    cv2.imshow('image', image)
    cv2.imshow('gaussian_noise', dst)
    cv2.waitKey(0)
