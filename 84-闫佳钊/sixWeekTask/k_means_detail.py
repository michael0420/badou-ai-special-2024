import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt


def function(img, k, channel):
    # img_gray = cv2.imread('lenna.png', 0)
    # img = cv2.imread('lenna.png', 0)
    h, w = img.shape[:2]
    # 簇数
    k = k
    # 质心坐标初始化，barycentric第一列是质心像素，第二列是质心簇所有坐标像素累加和，第三列是质心簇坐标数
    # new_img上次迭代图像每个坐标所属的簇索引，last_img本次迭代图像每个坐标所属的簇索引
    barycentric = np.zeros((k, 3, channel), dtype=np.uint64)
    new_img = np.zeros((h, w, channel), dtype=np.uint64)
    last_img = np.zeros((h, w, channel), dtype=np.uint64)
    # 质心像素初始化
    for i in range(k):
        barycentric[i, 0] = img[random.randint(0, h - 1), random.randint(0, w - 1)]

    for ch in range(channel):
        # 计算质心，count为迭代次数
        # flag = False
        count = 0
        while 1:
            count += 1
            for i in range(h):
                for j in range(w):
                    min_distance = 9999
                    min_distance_index = -1
                    for bary in range(k):
                        distance = abs(int(img[i, j, ch]) - int(barycentric[bary, 0, ch]))
                        if distance < min_distance:
                            min_distance = distance
                            min_distance_index = bary
                    last_img[i, j, ch] = min_distance_index
                    barycentric[min_distance_index, 1, ch] = int(barycentric[min_distance_index, 1, ch]) + int(img[i, j, ch])
                    barycentric[min_distance_index, 2, ch] = int(barycentric[min_distance_index, 2, ch]) + 1
            # 计算新的质心
            for bary in range(k):
                barycentric[bary, 0, ch] = 0 if barycentric[bary, 2, ch] == 0 else (barycentric[bary, 1, ch] / (barycentric[bary, 2, ch]))
                barycentric[bary, 1, ch] = 0
                barycentric[bary, 2, ch] = 0
            # 终止条件判断
            # last_img_ch = last_img[:, :, ch]
            # new_img_ch = new_img[:, :, ch]
            # new_img_ch == last_img_ch
            if (new_img == last_img).all() or count >= 10:
                if count >= 10:
                    new_img = last_img
                break
            new_img[:, :, ch] = last_img[:, :, ch]
    # 为不同簇的像素赋像素值
    for ch in range(channel):
        for i in range(h):
            for j in range(w):
                last_img[i][j][ch] = barycentric[last_img[i][j][ch], 0, ch]
    return last_img


if __name__ == '__main__':
    src_img = cv2.imread('lenna.png', -1)
    print(src_img)
    b, g, r = cv2.split(src_img)  # 分别提取B、G、R通道
    src_img = cv2.merge([r, g, b])  # 重新组合为R、G、B
    channel = 3
    plt_h = 1
    plt_w = 2
    plt.subplot(plt_h, plt_w, 1)
    plt.imshow(src_img, 'gray')
    plt.subplot(plt_h, plt_w, 2)
    plt.imshow(function(src_img, 2, channel))
    # plt.subplot(plt_h, plt_w, 3)
    # plt.imshow(function(src_img_gray, 8, channel))
    # plt.subplot(plt_h, plt_w, 4)
    # plt.imshow(function(src_img_gray, 64, channel))#, 'gray'
    plt.show()
