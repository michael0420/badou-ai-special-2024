import numpy as np
import cv2 as cv


def mean_hash(img):
    img = cv.resize(img, (8, 8), interpolation=cv.INTER_CUBIC)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    mean_hash_img = (gray > mean).astype('uint8')
    return np.array(mean_hash_img)


def Differential_hash(img):
    img = cv.resize(img, (9, 8), interpolation=cv.INTER_CUBIC)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    differential_hash_img = []
    for i in range(8):
        temp = []
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                temp.append(1)
            else:
                temp.append(0)
        differential_hash_img.append(temp)
    return np.array(differential_hash_img)


def perception_Hash(img, width=64, high=64):
    img = cv.resize(img, (width, high), interpolation=cv.INTER_CUBIC)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dct = cv.dct(gray.astype(dtype=np.float32))
    dct_real = dct.real
    resize_dct = np.resize(dct_real, (32, 32))
    mean = np.mean(resize_dct)
    perception_hash_img = np.array((resize_dct > mean).astype('int8'))
    perception_hash_img_flatten = ''.join(map(str, perception_hash_img.flatten()))
    # 每4位转成一个16进制
    return np.array(''.join([(''.join('%x' % (int(perception_hash_img_flatten[i:i + 4], 2))))
                    for i in range(0, len(perception_hash_img_flatten), 4)]))


img1 = cv.imread("../lenna.png")
img2 = cv.imread("../lenna_blurred.png")

mean_Hash_Img1 = mean_hash(img1)
mean_Hash_Img2 = mean_hash(img2)
differential_Hash_Img1 = Differential_hash(img1)
differential_Hash_Img2 = Differential_hash(img2)
perception_Hash1 = perception_Hash(img1)
perception_Hash2 = perception_Hash(img2)

mean_Similarity = (mean_Hash_Img1 != mean_Hash_Img2).astype('int8').sum()
differential_Similarity = (differential_Hash_Img1 != differential_Hash_Img2).astype('int8').sum()
perception_Similarity = (perception_Hash1 != perception_Hash2).astype('int8').sum()
print('均值哈希算法相似度：', mean_Similarity)
print('差值哈希算法相似度', differential_Similarity)
print('感知哈希算法相似度：', perception_Similarity)
