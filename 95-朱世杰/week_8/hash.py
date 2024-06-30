"""
实现差值哈希和均值哈希
@Author： zsj
"""

import cv2
import numpy as np


# 均值哈希值
def aHash(img):
    # 缩放图形至8*8
    dst = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # 求所有像素的平均值
    sum = np.sum(dst_gray)
    avg = sum / 64
    # 比较每个像素，大于平均值是1，反之为0
    hash = ''
    for i in range(8):
        for j in range(8):
            val = dst_gray[i][j]
            hash += '1' if val > avg else '0'
    return hash


# 差值哈希值
def dHash(img):
    # 缩放图形至8*8
    dst = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # 比较每个像素，大于后一个是1，反之为0
    hash = ''
    for i in range(8):
        for j in range(8):
            val = dst_gray[i][j]
            next_val = dst_gray[i][j + 1]
            hash += '1' if val > next_val else '0'
    return hash


# 比较哈希值
def compareHash(hash1, hash2):
    if len(hash1) != len(hash2):
        return -1
    xor = int(hash1, 2) ^ int(hash2, 2)
    xor_bin = bin(xor)
    return xor_bin.count('1')


img1 = cv2.imread('./example/source/lenna.png')
img2 = cv2.imread('./example/source/lenna_rotate.jpg')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = compareHash(hash1, hash2)
print('均值哈希算法相似度：', n)

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = compareHash(hash1, hash2)
print('差值哈希算法相似度：', n)
