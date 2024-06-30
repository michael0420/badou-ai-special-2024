import cv2
import numpy as np


# 均值哈希算法
def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
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


# 差值算法
def dHash(img):
    # 缩放8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


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



import numpy as np
import cv2
from numpy import shape
import random
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        if NoiseImg[randX,randY]<0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX,randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg
img1 = cv2.imread('lenna.png')
img2 = GaussianNoise(img1,2,4,0.8)
cv2.imwrite('lenna.png_noisy.png', img1*255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

cv2.waitKey(0)
img3 = cv2.imread('lenna.png_noisy.png')
hash1 = aHash(img1)
hash2 = aHash(img3)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)

hash1 = dHash(img1)
hash2 = dHash(img3)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', n)