import cv2
import numpy as np


# 均值哈希
def mHash(img, width=8, high=8):
    """
        均值哈希算法
        :param img: 图像数据
        :param width: 图像缩放的宽度
        :param high: 图像缩放的高度
        :return:感知哈希序列
        """
    # 缩放为8*8
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

#两张图Hash值对比
def cmpHash(hash1, hash2):
    """
        Hash值对比
        :param hash1: 感知哈希序列1
        :param hash2: 感知哈希序列2
        :return: 返回不相似的像素个数，返回相似度
        """
    n = 0
    #hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n+1
    return n
    return 1 - n / len(hash2)

img1 = cv2.imread('image_all/lenna.png')
img2 = cv2.imread('lenna_noise.png')
hash1 = mHash(img1)
hash2 = mHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)
print('均值哈希算法相似度：', 1 - n / len(hash2))