# coding = utf-8

'''
        双线性插值
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

def bilinear_interpolation(img, out):
    s_h, s_w, s_c = img.shape
    f_h, f_w = out[1], out[0]
    # 判断：尺寸相等则复制返回
    if s_h == f_h and s_w == f_w:
        return img.copy()

    # 创建空图片对象
    final_pics = np.zeros((f_h, f_w, 3), dtype=np.uint8)

    # 计算比例
    ph = float(s_h / f_h)
    pw = float(s_w / f_w)

    for i in range(s_c):    # 通道数循环
        for j in range(f_h):
            for k in range(f_w):

                # 中心对称
                x = (k + 0.5) * pw - 0.5
                y = (j + 0.5) * ph - 0.5

                # 防止越界
                s_h0 = int(np.floor(y))
                s_h1 = min(s_h0 + 1, s_h - 1)
                s_w0 = int(np.floor(x))
                s_w1 = min(s_w0 + 1, s_w - 1)

                # 双线性插值计算 (x1-x)*像素值 + (x-x0)*像素值???
                # x方向
                # p1 = (s_w1 - x) * img[s_h0, s_w0, i] + (x - s_w0) * img[s_h0, s_w1, i]
                # p2 = (s_w1 - x) * img[s_h1, s_w0, i] + (x - s_w0) * img[s_h1, s_w1, i]
                # final_pics[j, k, i] = int((s_h1 - y) * p1 + (y - s_h0) * p2)

                # y方向
                p3 = (s_h1 - y) * img[s_h0, s_w0, i] + (y - s_h0) * img[s_h0, s_w1, i]
                p4 = (s_h1 - y) * img[s_h1, s_w0, i] + (y - s_h0) * img[s_h1, s_w1, i]
                final_pics[j, k, i] = int((s_w1 - x) * p3 + (x - s_w0) * p4)

    return final_pics


img = cv2.imread('lenna.png')
cv2.imshow('src', img)

res = bilinear_interpolation(img, (1000, 1000))
cv2.imshow('target', res)
cv2.waitKey(0)
