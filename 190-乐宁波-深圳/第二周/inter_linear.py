import cv2
import numpy as np

original_img = cv2.imread('img.jpg')

scale_factor = 2  # 缩放系数，如：放大2倍

h, w, c = original_img.shape
new_h, new_w = int(h * scale_factor), int(w * scale_factor)

new_img = np.zeros([new_h, new_w, c], dtype=original_img.dtype)

for y in range(new_h):
    for x in range(new_w):
        nx = (x + 0.5) / scale_factor - 0.5  # 中心对齐
        ny = (y + 0.5) / scale_factor - 0.5

        # nx = x / scale_factor  # opencv用的是这个方式
        # ny = y / scale_factor

        u = nx % 1
        v = ny % 1

        Q11 = int(ny), int(nx)
        Q12 = int(ny), min(int(nx) + 1, w - 1)
        Q21 = min(int(ny) + 1, h - 1), int(nx)
        Q22 = min(int(ny) + 1, h - 1), min(int(nx) + 1, w - 1)

        R = (1 - u) * (original_img[Q11]) + u * (original_img[Q12])
        S = (1 - u) * (original_img[Q21]) + u * (original_img[Q22])
        T = (1 - v) * R + v * S

        new_img[y, x] = T

cv2.imshow('inter linear img1', new_img)
resized_image = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # 对比opencv的双线性插值结果
cv2.imshow('inter linear img2', resized_image)
print(new_img)
print('--------------------------------------')
print(resized_image)
cv2.waitKey(0)
