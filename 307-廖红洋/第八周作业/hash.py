# -*- coding: utf-8 -*-
"""
@File    :   hash.py
@Time    :   2024/06/16 17:14:57
@Author  :   廖红洋 
"""
import cv2
import numpy as np

imgo = cv2.imread("lenna.png")
imgo2 = cv2.imread("lenna_noise.png")
# 均值哈希
img = cv2.resize(imgo, (8, 8), interpolation=cv2.INTER_CUBIC)  # resize方法，对图片进行缩放
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 生成灰度图
img2 = cv2.resize(imgo2, (8, 8), interpolation=cv2.INTER_CUBIC)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
s = 0
hash_str = ""
hash_str2 = ""
for i in gray:
    for j in i:
        s += j
s = s / 64
for i in gray:
    for j in i:
        if j > s:
            hash_str = hash_str + "1"
        else:
            hash_str = hash_str + "0"
hash_str2 = ""
for i in gray2:
    for j in i:
        s += j
s = s / 64
for i in gray2:
    for j in i:
        if j > s:
            hash_str2 = hash_str2 + "1"
        else:
            hash_str2 = hash_str2 + "0"
# 差值哈希
img3 = cv2.resize(imgo, (8, 9), interpolation=cv2.INTER_CUBIC)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4 = cv2.resize(imgo2, (8, 9), interpolation=cv2.INTER_CUBIC)
gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
hash_str3 = ""
hash_str4 = ""
for i in range(8):
    for j in range(8):
        if gray3[i + 1][j] > gray3[i][j]:
            hash_str3 = hash_str3 + "1"
        else:
            hash_str3 = hash_str3 + "0"
for i in range(8):
    for j in range(8):
        if gray4[i + 1][j] > gray4[i][j]:
            hash_str4 = hash_str4 + "1"
        else:
            hash_str4 = hash_str4 + "0"
print("均值哈希" + "\n")
print("原图：" + hash_str + "\n")
print("噪声：" + hash_str2 + "\n")
print("差值哈希" + "\n")
print("原图：" + hash_str3 + "\n")
print("噪声：" + hash_str4)
