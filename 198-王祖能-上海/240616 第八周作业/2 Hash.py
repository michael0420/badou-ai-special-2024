'''
图像相似度处理：均值哈希，插值哈希
'''
import cv2
import numpy as np
import time


def aHash(src):  # 均值哈希算法：转换为8*8图片，求灰度平均值，各点灰度大于平均值为1，相反为0，形成hash值
    src = cv2.resize(src, [8, 8], interpolation=cv2.INTER_CUBIC)  # dsize=[0, 0], fx=1.5, fy=1.2按比例放大，尺寸自动计算
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ave = np.sum(src) / 64
    ahash = ''
    for i in range(8):
        for j in range(8):
            if src[i, j] > ave:
                ahash += '1'
            else:
                ahash += '0'
    return ahash


def dHash(src):  # 差值哈希算法：转换为8*9图片，每行独立不与其他行比，像素值比后一位置大为1，相反为0，形成hash值
    src = cv2.resize(src, [9, 8], interpolation=cv2.INTER_CUBIC)  # resize参数要求先列后行,要改为（9， 8），三次样条插值图像缩放
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dhash = ''
    for i in range(8):
        for j in range(8):
            if src[i, j] > src[i, j + 1]:
                dhash += '1'
            else:
                dhash += '0'
    return dhash


def cmpHash(hash1, hash2):
    len1, len2 = len(hash1), len(hash2)
    num = 0
    if len1 != len2:
        print('无法比较')
    for i in range(len1):
        if hash1[i] == hash2[i]:
            num += 1
    percent = num / len1
    return percent


img1 = cv2.imread('lenna.png')
img2 = cv2.GaussianBlur(img1, ksize=[5, 5], sigmaX=15)
start1 = time.time()
a1, a2 = aHash(img1), aHash(img2)
per1 = cmpHash(a1, a2)
end1 = time.time()
time1 = end1 - start1

start2 = time.time()
d1, d2 = dHash(img1), dHash(img2)
per2 = cmpHash(d1, d2)
end2 = time.time()
time2 = end2 - start2
print('均值哈希的汉明距离结果：\n图1为:{}\n图2为:{}，图像相似度为:{:.4f}，计算耗时:{:.8f}'.format(a1, a2, per1, time1))
print('差值哈希的汉明距离结果：\n图1为:{}\n图2为:{}，图像相似度为:{:.4f}，计算耗时:{:.8f}'.format(d1, d2, per2, time2))
cv2.imshow('contrast', np.hstack([img1, img2]))
cv2.waitKey()
cv2.destroyAllWindows()
