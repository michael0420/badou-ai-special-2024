import cv2
import numpy as np


# 均值哈希算法
def aHash(img):
    # 缩放为8*8
    """
    1.缩放成 8X8, 保留结构，除去细节。
      将图像 `img` 调整为大小为 8x8 的新图像，并使用了三次样条插值方法（cv2.INTER_CUBIC）来保持图像的平滑性。
      三次样条插值是一种插值方法，它通过拟合一个三次多项式函数来生成新的像素值。这种方法可以在保持图像细节的同时，有效地减少图像的锯齿和失真。
    """
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    '''
    2.转换为灰度图
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''
    3.计算灰度图所有像素的平均值
    '''
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 64
    '''
    4.比较:像素值大于平均值记作1，相反记作0，总共64位
    5.生成hash:将上述步骤生成的1和0按顺序组合起来既是图片的指纹
    '''
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值算法
def dHash(img):
    # 1.缩放8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 2.转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 4.每行前一个像素大于后一个像素为1，相反为0，生成哈希
    #   本行不与下一行对比，每行9个像素，8个插值，有8行，总共64位
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


'''6.对比指纹:将两幅图的指纹比较，计算汉明距离，即两个64位hash值有多少位是不一样的，不相同位数越少，图片越相似'''
# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


img1 = cv2.imread('D:\cv_workspace\picture\lenna.png')
img2 = cv2.imread('D:\cv_workspace\picture\lenna_noise.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)
'''
1011111010011110100111011010100110101011101000110000111000101100
1011111010111110100111011000100110101011101000110000101000101100
均值哈希算法相似度： 3
'''

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', n)
'''
1000100110001101101000101010010001000110111011001010010110000011
1010100110001101101000101010010001000110011011000010010110000011
差值哈希算法相似度： 3
'''
