# coding = utf-8

"""
        实现均值哈希
"""

import numpy as np
import cv2
from PIL import Image


def ave_hash(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    todo_img = cv2.resize(img_gray, (8, 8), interpolation=cv2.INTER_CUBIC)  # INTER_CUBIC重采样方法
    sum = 0
    ahash = ''
    for i in range(todo_img.shape[0]):
        for j in range(todo_img.shape[1]):
            sum += todo_img[i, j]

    ave_pixel = sum / 64
    for i in range(todo_img.shape[0]):
        for j in range(todo_img.shape[1]):
            if todo_img[i, j] > ave_pixel:
                ahash = ahash + '1'
            else:
                ahash = ahash +'0'
    return ahash

def compare(str1, str2):
    diff = 0
    # 判断哈希长度是否相等
    if len(str1) != len(str2):
        diff = -1
    else:
        for i in range(len(str1)):
            if str1[i] != str2[i]:
                diff += 1

    return diff


if __name__ == '__main__':
    lenna = cv2.imread('lenna.png')
    lenna_blur = cv2.blur(lenna, (5, 5))    # 模糊化
    # cv2.imshow('lenna', lenna)
    # cv2.imshow('blur', lenna_blur)


    hash_1 = ave_hash(lenna)
    hash_2 = ave_hash(lenna_blur)
    diff = compare(hash_1, hash_2)
    print('图片的汉明距离差异数：', diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
