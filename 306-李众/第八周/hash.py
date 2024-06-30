import cv2
import numpy as np

#均值哈希
def aHash(img):
    img = cv2.resize(img,(8,8))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    hashstr = ''
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i][j] > mean:
                hashstr += '1'
            else:
                hashstr += '0'
    return hashstr

#差值哈希
def dHash(img):
    img = cv2.resize(img,(8,9))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hashstr = ''
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]-1):
            if gray[i][j] > gray[i][j+1]:
                hashstr += '1'
            else:
                hashstr += '0'
    return hashstr

#对比哈希值
def cmpHash(str1,str2):
    d = 0
    if len(str1) != len(str2) :
        return -1
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            d += 1
    return d


img1 = cv2.imread('img1.png')
img2 = cv2.imread('img2.png')
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
