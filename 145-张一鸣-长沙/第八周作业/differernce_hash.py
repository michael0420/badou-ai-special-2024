# coding = utf-8

'''
        实现差值哈希
'''

import numpy as np
import cv2


def diff_hash(img):
    todo_img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(todo_img, cv2.COLOR_BGR2GRAY)
    bhash = ''
    print(img_gray.shape)
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[0]):
            if img_gray[i, j] > img_gray[i, j+1]:
                bhash = bhash + '1'
            else:
                bhash = bhash + '0'
    return bhash

def compare(str1, str2):
    diff = 0
    # 判断哈希长度是否相等
    if len(str1) != len(str2):
        diff = -1
    else:
        for i in range(len(str1)):
            if str1[i] != str2[i]:
                diff += 1

    return diff

if __name__ == '__main__':
    if __name__ == '__main__':
        lenna = cv2.imread('lenna.png')
        # np.clip()限定数值在某个范围内，np,uint8()转为8位无符号整数
        lenna_light = np.uint8(np.clip((lenna * 2 + 15), 0, 255))  # 调整亮度对比度
        # cv2.imshow('lenna', lenna)
        # cv2.imshow('light', lenna_light)

        hash_1 = diff_hash(lenna)
        hash_2 = diff_hash(lenna_light)
        diff = compare(hash_1, hash_2)
        print('图片的汉明距离差异数：', diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
