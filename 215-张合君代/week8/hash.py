# -*- coding: utf-8 -*-
"""
@author: zhjd

aHash dHash image comparison
"""
import glob
import itertools
import os

import cv2
import numpy as np


def ahash(image, hash_size=(8, 8)):
    resize_image = cv2.resize(image, hash_size, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    # 计算灰度图像的像素平均值
    mean_value = np.mean(gray)
    height, width = gray.shape
    hash_code = ''
    for i in range(height):
        for j in range(width):
            hash_code += '1' if gray[i, j] > mean_value else '0'

    return hash_code


def dhash(image, hash_size=(8, 9)):
    resize_image = cv2.resize(image, hash_size, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    hash_code = ''
    for i in range(height - 1):
        for j in range(width):
            hash_code += '1' if gray[i, j] > gray[i + 1, j] else '0'

    return hash_code


def hamming_distance(hash_code1, hash_code2):
    if len(hash_code1) != len(hash_code2):
        return -1

    distance = 0
    for i in range(len(hash_code1)):
        if hash_code1[i] != hash_code2[i]:
            distance += 1

    return distance


if __name__ == '__main__':
    # cv2.resize参数interpolation的验证，缩小图像时，使用INTER_CUBIC汉明距离大，使用INTER_AREA汉明距离小
    directory_path = 'source'
    file_names = glob.glob(os.path.join(directory_path, '*'))
    compare_list = list(itertools.combinations(file_names, 2))
    for i in compare_list:
        image1 = cv2.imread(i[0])
        image2 = cv2.imread(i[1])
        print(str(i) + ' compare by ahash: ' + str(hamming_distance(ahash(image1), ahash(image2))))
        print(str(i) + ' compare by dhash: ' + str(hamming_distance(dhash(image1), dhash(image2))))
