import cv2
from skimage import util

# 均值哈希算法
def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_sum = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            hash_sum = hash_sum + img_gray[i, j]
    avg = hash_sum / 64
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值算法
def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > img_gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# Hash值对比
def cmpHash(hash1, hash2):
    count = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            count = count + 1
    return count


# 增加噪声
def zaosheng(img):
    img_noise = util.random_noise(img, mode='s&p', amount=0.0001)
    cv2.imshow("d", img_noise)
    cv2.waitKey(3000)
    cv2.imwrite('lenna_noise.png',img_noise*255)


img1 = cv2.imread('lenna.png')
zaosheng(img1)
img2 = cv2.imread('lenna_noise.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
count = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', count)

hash1 = dHash(img1)
hash2 = dHash(img2)
count = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', count)