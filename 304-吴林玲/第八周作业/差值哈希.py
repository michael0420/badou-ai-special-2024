import cv2
import numpy as np
import time
import os.path as path



#实现差值哈希算法
def dHash(img, width=9, high=8):
    # 缩放8*8
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，反之置为0，生成感知哈希序列（string）
    for i in range(high):
        for j in range(high):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str