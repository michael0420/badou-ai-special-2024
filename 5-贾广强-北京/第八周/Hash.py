import cv2
import numpy as np
import skimage.util as ut

def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i,j]
    avg = s/64

    for i in range(8):
        for j in range(8):
            if gray[i,j] >avg:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str

def dHash(img):
    img = cv2.resize(img, (9,8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

def camHash(hash1,hash2):
    n = 0
    if len(hash1)!= len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n = n+1
    return n

img1= cv2.imread('lenna.jpg')
noise_img = ut.random_noise(img1, mode='s&p')
img2=(np.clip(noise_img,0,1) * 255).astype(np.uint8)

hash1 = aHash(img1)
hash2 = aHash(img2)
n1 = camHash(hash1, hash2)
print('原始图像均值哈希值：', hash1)
print('噪声图像均值哈希值：', hash2)
print('均值哈希相似度', n1)

hash1= dHash(img1)
hash2= dHash(img2)
n2 = camHash(hash1, hash2)
print('原始图像差值哈希值：', hash1)
print('噪声图像差值哈希值：', hash2)
print('差值哈希相似度', n2)

