import cv2
import numpy as np


def aHash(img):
    dst = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    adst = 0
    for i in range(8):
        for j in range(8):
            adst += dst_gray[i, j]
    adst /= 64
    # adst = sum(dst_gray) / 64
    ahash_str = ''
    for i in range(8):
        for j in range(8):
            if dst_gray[i, j] > adst:
                ahash_str += '1'
            else:
                ahash_str += '0'
    return ahash_str


def dHash(img):
    dst = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ahash_str = ''
    for i in range(8):
        for j in range(8):
            if dst_gray[i, j] > dst_gray[i, j + 1]:
                dst_gray[i, j] = 1
            else:
                dst_gray[i, j] = 0
            ahash_str += str(dst_gray[i, j])
    return ahash_str


def han_ming(img1, img2):
    if len(img1) != len(img2):
        return -1
    count = 0
    for i in range(len(img1)):
        if img1[i] != img2[i]:
            count += 1
    return count


if __name__ == '__main__':
    src1 = cv2.imread('../lenna.png')
    src2 = cv2.imread('../lenna_noise.png')
    aHash_str1 = aHash(src1)
    aHash_str2 = aHash(src2)
    aHash_count = han_ming(aHash_str1, aHash_str2)
    dHash_str1 = dHash(src1)
    dHash_str2 = dHash(src2)
    dHash_count = han_ming(dHash_str1, dHash_str2)
    print('aHash_count=', aHash_count)
    print('dHash_count=', dHash_count)
