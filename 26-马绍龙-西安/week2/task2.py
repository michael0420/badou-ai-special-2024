# 2.实现最临近插值
import cv2
import numpy as np

def zoom(img,height,width):
    h_old, w_old, channels = img.shape[0:3]
    zoom_result = np.zeros((height, width, channels), np.uint8)
    times_h = height / h_old
    times_w = width / w_old
    for i in range(height):
        for j in range(width):
            x = int(i/times_h+0.5)   #int(),转为整型，使用向下取整。
            y = int(j/times_w+0.5)
            if 0 <= x < h_old and 0 <= y < w_old:
                zoom_result[i, j] = img[x, y]
    return zoom_result

img = cv2.imread('./lenna.png')
zoom_result = zoom(img,600,600)
cv2.imshow('img', img)
cv2.imshow('zoom_result', zoom_result)
cv2.waitKey(0)

