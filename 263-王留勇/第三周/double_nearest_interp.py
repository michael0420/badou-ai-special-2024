'''
双线性插值
'''

import cv2
import numpy as np

def function(img):
    height, width, channel = img.shape
    print('height=%d, width=%d' % (height, width))
    emptyImage = np.zeros((800, 800, 3), dtype=np.uint8)
    sh = height/800
    sw = width/800
    for i in range(3):
        for dst_y in range(800):
            for dst_x in range(800):
                src_x = (dst_x + 0.5) * sw - 0.5
                src_y = (dst_y + 0.5) * sh - 0.5
                # 找到周围四个像素的坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, width - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, height - 1)
                # 双线性插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                # 将计算得到的像素值放入新图像
                emptyImage[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return emptyImage

img = cv2.imread('lenna.png')
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow('nearest interp', zoom)
cv2.imshow('image', img)
cv2.waitKey(0)
