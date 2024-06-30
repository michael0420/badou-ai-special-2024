import cv2
import numpy as np

"""
1. 缩放：图片缩放为8*8，保留结构，除去细节
2. 灰度化：转换为灰度图
3. 求平均值：计算灰度图所有像素的平均值
4. 比较：像素值大于平均值记作1，相反记作0，总共64位
5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）
6. 对比指纹：将两幅图的指纹对比，计算汉明距离;两个64位的hash值有多少位是不一样的，不相同位数越少，图片越相似
"""
# 均值哈希
def avgHash(img):
    # 1.缩放为8*8
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    # 2.转换为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 3.Sum：像素值求和，初始值为0, 输出hash值为：hash_str
    Sum = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            Sum = Sum + gray[i,j]
    # 求平均值
    avg = Sum/64
    # 4.比较，灰度值大于平均值为1，反之则为0，生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# hash值对比
def cmpHash(hash1,hash2):
    # 两个hash值的原始差异数为0
    n = 0
    if len(hash1)!=len(hash2):
        return -1
    # 遍历判断,同位数字不相等则+1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n+1
    return n

img1 = cv2.imread('lenna.png')
hash1 = avgHash(img1)
print("原图hash值：",hash1)
img2 = cv2.imread('noise_gs_img.png')
hash2 = avgHash(img2)
print("噪声hash值：",hash2)

n = cmpHash(hash1,hash2)
print("均值哈希算法相似度：",n)