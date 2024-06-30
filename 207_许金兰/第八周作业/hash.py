"""
@author: 207-xujinlan
实现均值哈希和差值哈希算法
"""

import cv2
import numpy as np


def avg_hash(img):
    """
    均值哈希算法
    :param img: 输入图片
    :return: 返回均值哈希字符串
    """
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)  # 图片缩放
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片灰度化
    m = img_gray.mean()  # 求平均值
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] >= m:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


def diff_hash(img):
    """
    差值哈希算法
    :param img:输入图片
    :return: 返回差值哈希字符串
    """
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)  # 图片缩放
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片灰度化
    hash_str = ''
    for i in range(8):
        for j in np.arange(1, 9):
            if img_gray[i, j] >= img_gray[i, j - 1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    ahash = avg_hash(img)
    dhash = diff_hash(img)
