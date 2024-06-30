# _*_ coding: UTF-8 _*_
# @Time: 2024/4/17 15:35
# @Author: iris
import random

import cv2


def salt_pepper(src, ratio):
    """
    高斯造册亨
    :param image:
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
        if random.random() <= 0.5:
            dst_image[src_x, src_y] = 255
        elif random.random() > 0.5:
            dst_image[src_x, src_y] = 0

    return dst_image


if __name__ == '__main__':
    image = cv2.imread('../data/lenna.png')
    dst = salt_pepper(image, 0.05)
    image = cv2.imread('../data/lenna.png')
    cv2.imshow('image', image)
    cv2.imshow('salt_noise', dst)
    cv2.waitKey(0)
