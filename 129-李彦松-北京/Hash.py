import cv2
import numpy as np

# 均值哈希算法
def aHash(img):
    # 缩放为8*8并转换为灰度图
    gray = cv2.cvtColor(cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    # 计算平均灰度
    avg = np.mean(gray)
    # 灰度大于平均值为1，相反为0，生成图片的哈希值
    hash_str = ''.join(['1' if i > avg else '0' for i in gray.flatten()]) ## flatten()将数组变为一维，''.join([...])是将生成的字符串列表连接成一个字符串。
    return hash_str

# 差值哈希算法
def dHash(img):
    # 缩放8*9并转换为灰度图
    gray = cv2.cvtColor(cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    hash_str = ''.join(['1' if gray[i, j] > gray[i, j + 1] else '0' for i in range(8) for j in range(8)])
    return hash_str

# Hash值对比
def cmpHash(hash1, hash2):
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 计算两个哈希值的差异
    return np.sum(np.array(list(hash1)) != np.array(list(hash2)))

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', n)