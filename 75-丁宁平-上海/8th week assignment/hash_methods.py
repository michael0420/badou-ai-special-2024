# coding=utf-8


import cv2
import numpy as np


# 均值哈希算法
def aHash(img):
    '''
    均值哈希算法
    :param img:图像
    :return:哈希值
    '''
    # 图像缩放为8*8px
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为初始化像素值总和为0；hash_str为哈希值，初始值为''
    s = 0
    hash_str = ''

    # 遍历累加求像素总和
    for i in range(8):
        for j in range(8):
            s += gray[i,j]
    # 求平均灰度值
    avg = s/(8*8)
    # 灰度值对于平均灰度值的记为1，小于平均灰度值的记为0
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


def dHash(img):
    '''
    差值哈希算法
    :param img:原图
    :return: 哈希值
    '''
    # 图像缩放为8*9px(差值相减需要多1)
    img = cv2.resize(img, (9, 8),interpolation=cv2.INTER_CUBIC)  # (width,height)
    # 转为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 初始化hash_str值为0
    hash_str = ''

    # 每行前一个像素值大于后一个像素值的记为1，反之记为0
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


# 哈希值对比
def cmpHash(hash1, hash2):
    '''
    汉明距离：比较2个二进制数位不相同的数量
    :param hash1:
    :param hash2:
    :return:
    '''
    # 初始化差值为0
    dn = 0
    # hash值数量不同无需比较，返回-1
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 位不相同则n++，n为最终相似度
        if hash1[i] != hash2[i]:
            dn += 1
    return dn


'''--------------------------------- 【均值哈希结果】---------------------------------'''
img1 = cv2.imread('lenna.png')
# img2 = cv2.imread('lenna.png')
img2 = cv2.imread('cmp_img')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print(f'均值哈希算法相似度为：{n}')


'''--------------------------------- 【差值哈希结果】---------------------------------'''
hash3 = dHash(img1)
hash4 = dHash(img2)
print(hash3)
print(hash4)
n1 = cmpHash(hash3,hash4)
print(f'差值哈希算法相似度为：{n1}')