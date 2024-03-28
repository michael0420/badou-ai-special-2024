# 邻近插值法
import cv2
import numpy as np

img = cv2.imread('lenna.png')
h, w, channels = img.shape

new_img = np.zeros((800, 800, channels), np.uint8)

# 获取比例
sh = 800 / h
sw = 800 / w

# 大图的(x, y)在小图对应的就是 (x/sw, y/sh)

for i in range(800):
    for j in range(800):
        # 坐标不可能是小数，所以一定要整数化处理
        x = int(i / sw + 0.5) # 代替四舍五入的方法
        y = int(j / sh + 0.5)
        new_img[i, j] = img[x, y]

cv2.imshow('new_img', new_img)
cv2.waitKey(0)