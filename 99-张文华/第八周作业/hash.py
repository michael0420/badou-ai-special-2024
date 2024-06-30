'''
1、均值hash
2、差值hash
'''

import cv2
import numpy as np


# 定义一个均值hash方法
def aHash(img):
    img_resize = cv2.resize(img, (8, 8))
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    mean = np.sum(img_resize) / 64
    # print(mean)
    img_resize = img_resize.reshape((64, 1))
    H = ''
    for i in img_resize:
        if i >= mean:
            H += '1'
        else:
            H += '0'
    return H


# 定义一个插值hash的方法
def dHash(img):
    img_resize = cv2.resize(img, (9, 8))
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    H = ''
    for i in range(8):
        for j in range(8):
            if img_resize[i, j] >= img_resize[i, j+1]:
                H += '1'
            else:
                H += '0'
    return H


# 定义一个方法求汉明距离
def distance(h1, h2):
    if len(h1) != len(h2):
        print('输入hash位数不一致，无法求汉明距离')
        return
    D = 0
    for i in range(len(h1)):
        if h1[i] != h2[i]:
            D += 1
    return D

# 定义一个方法给图片加入高斯噪声
def noise(img):
    gauss_noise = np.random.normal(size=img.shape).astype('uint8')
    img_noise = cv2.add(img, gauss_noise)
    return img_noise


if __name__ == '__main__':
    img_src = cv2.imread('lenna.png')
    img_gauss = noise(img_src)
    img = np.hstack((img_src,img_gauss))
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyWindow('img')
    # 使用均值hash看两图差异
    img_src_ahash = aHash(img_src)
    img_gauss_ahash = aHash(img_gauss)
    distance_ahash = distance(img_gauss_ahash, img_src_ahash)
    print(f'均值HASH中\n原始图的hash值是{img_src_ahash}\n'
          f'噪声图的hash值是{img_gauss_ahash}\n汉明距离={distance_ahash}')
    # 使用差值hash看两图差异
    img_src_dhash = dHash(img_src)
    img_gauss_dhash = dHash(img_gauss)
    distance_dhash = distance(img_gauss_dhash, img_src_dhash)
    print(f'差值HASH中\n原始图的hash值是{img_src_dhash}\n'
          f'噪声图的hash值是{img_gauss_dhash}\n汉明距离={distance_dhash}')
