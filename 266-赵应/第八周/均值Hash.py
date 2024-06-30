import cv2
import numpy as np


def mean_hash(img):
    """计算均值hash"""
    img_scale = cv2.resize(img, (8, 8), interpolation=cv2.INTER_NEAREST)
    img_gray = cv2.cvtColor(img_scale, cv2.COLOR_BGR2GRAY)
    img_average = int(np.mean(img_gray))
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i][j] > img_average:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


def cal_hanming_distance(s1, s2):
    """计算汉明距离"""
    if len(s2) != len(s1):
        raise ValueError('Length of strings must be equal!')
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


if __name__ == '__main__':
    img1 = cv2.imread("./image/lenna.png")
    img2 = cv2.imread("./image/lenna.png")
    hash_str1 = mean_hash(img1)
    hash_str2 = mean_hash(img2)
    print(hash_str1)
    print(hash_str2)
    print(cal_hanming_distance(hash_str1, hash_str2))




