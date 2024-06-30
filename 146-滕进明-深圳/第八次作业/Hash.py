import cv2
import numpy as np


# 均值哈希算法
def aHash(img):
    # 图片尺寸缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)  # interpolation=cv2.INTER_CUBIC是OpenCV库中用于图像缩放的一种插值方法。它使用三次样条插值来重新采样图像像素
    # 将缩放的图片转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 设图片像素和值初始为0，hash_str为hash值初值为''
    s=0
    hash_str = ''
    # 遍历缩放图片的像素，求像素和值
    for i in range(len(gray)):
        for j in range(len(gray[i])):
            s += gray[i, j]

    # 求平均灰度值
    avg = s/(len(gray) * len(gray[0]))

    # 遍历缩放灰度图片，灰度值大于平均值哈希值+1，灰度值小于平均值哈希值+0
    for i in range(len(gray)):
        for j in range(len(gray[i])):
            if gray[i, j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

# 差值算法
def dHash(img):
    # 缩放图片尺寸为宽9 * 高8
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_NEAREST)
    # 图像转化为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 哈希字符串初始化
    hash_str = ''
    # 每行前一个像素大于后一个像素为1， 相反为0，生成哈希
    for i in range(len(gray)):
        for j in range(len(gray[i]) - 1):
            if gray[i, j] > gray[i, j + 1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n1 = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n1)

hash3 = dHash(img1)
hash4 = dHash(img2)
print(hash3)
print(hash4)
n2 = cmpHash(hash3, hash4)
print('差值哈希算法相似度：', n2)







