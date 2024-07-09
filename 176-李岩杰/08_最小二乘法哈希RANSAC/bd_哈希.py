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


img1 = cv2.imread('lenna.png')
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# 定义高斯噪声参数
mean = 0  # 均值
var = 20  # 方差，控制噪声强度
sigma = var**0.5  # 标准差

# 生成高斯噪声
noise = np.random.normal(mean, sigma, img1.shape)

# 将噪声添加到图片
noisy_img = img1 + noise

# 确保像素值在0-255之间
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# 显示原图和加噪后的图片
cv2.imshow('Original Image', img1)
cv2.imshow('Noisy Image', noisy_img)
cv2.imwrite('noisy_image.png', noisy_img)
# 等待按键后关闭所有窗口


img2 = cv2.imread('noisy_image.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法有%s位不一样' %n)

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法有%s位不一样' %n)
cv2.waitKey(0)