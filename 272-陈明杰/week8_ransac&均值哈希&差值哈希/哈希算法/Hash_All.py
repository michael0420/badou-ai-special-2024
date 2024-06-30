import cv2
import numpy as np

# 哈希算法可以根据汉明距离求相似度
# 均值哈希算法
def avgHash(src):
    Sum = 0
    # 把原图像缩放为8*8大小
    # cv2.INTER_NEAREST：最近邻插值
    # cv2.INTER_LINEAR：双线性插值（默认）
    # cv2.INTER_AREA：使用像素区域关系进行重采样。当图像被缩小时，它可能会避免波纹出现。
    # cv2.INTER_CUBIC：双三次插值（对于放大图像，它可能产生更平滑的边缘）
    # cv2.INTER_LANCZOS4：Lanczos插值
    src = cv2.resize(src, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 记得要转成灰度图
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    for i in range(8):
        for j in range(8):
            Sum += src[i, j]

    avg = Sum / 64
    res = ''

    for i in range(8):
        for j in range(8):
            if src[i, j] > avg:
                res += '1'
            else:
                res += '0'
    return res


def subHash(src):
    # 这里的第二个参数是(width,high)格式的，所以我们要缩放为8*9的矩阵，填写的参数是(9,8)
    # cv2.INTER_NEAREST：最近邻插值
    # cv2.INTER_LINEAR：双线性插值（默认）
    # cv2.INTER_AREA：使用像素区域关系进行重采样。当图像被缩小时，它可能会避免波纹出现。
    # cv2.INTER_CUBIC：双三次插值（对于放大图像，它可能产生更平滑的边缘）
    # cv2.INTER_LANCZOS4：Lanczos插值
    src = cv2.resize(src, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 记得要转成灰度图
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    res = ''
    for i in range(8):
        for j in range(8):
            if src[i, j] < src[i, j + 1]:
                res += '1'
            else:
                res += '0'
    return res


# Hash值对比
def hashCmp(hash1, hash2):
    if len(hash1) != len(hash2):
        return 0
    n = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    # return n/len(hash1)
    return n

