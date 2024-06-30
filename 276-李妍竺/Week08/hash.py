import cv2
import numpy as np


# 均值哈希算法
def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_LINEAR)
    '''
    INTER_NEAREST	最邻近插值	              0
    INTER_LINEAR	双线性插值 （默认）	      1
    INTER_CUBIC	    4x4像素邻域内的双立方插值	  2
    INTER_AREA	  使用像素区域关系进行重采样	  3
    INTER_LANCZOS4 8x8像素邻域内的Lanczos插值  4
    '''
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和 初值为0，hash_str为hash值(0,1)初值为''
    s = 0
    hash_str = ''

    s = np.sum(gray)
    '''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]  # 共计64个像素值
    '''
    # 求平均灰度
    avg = s / 64

    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'    # 累积的，按字符串算1，0
    return hash_str

# 差值哈希算法

def dHash(img):
    # 缩放8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)  # 这里的（9，8）指x方向，y方向
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:  # 相同的i, 不进行跨行比较
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# Hash值对比
def cpHash(hash1, hash2):

    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    n = 0
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1

    per = 1 - n / len(hash2)  #输出相似比例
    return n, per

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n_per = cpHash(hash1, hash2)

print('均值哈希算法相似度：',n_per)


hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n_per = cpHash(hash1, hash2)
print('差值哈希算法相似度：', n_per)
