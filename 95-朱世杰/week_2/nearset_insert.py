"""
最邻近插值
@author: zsj
"""
import cv2
import numpy as np


# 最邻近插值
def nearestInsertValue(img, height, width):
    old_height, old_width, c = img.shape
    h_ratio = height / old_height
    w_ratio = width / old_width
    new_img = np.zeros([height, width, c], img.dtype)
    for i in range(height):
        for j in range(width):
            row = min(round(i / h_ratio), old_height - 1)
            # row = round(i / h_ratio)
            col = min(round(j / w_ratio), old_width - 1)
            # col = round(j / w_ratio)
            new_img[i][j] = img[row][col]
    return new_img


img = cv2.imread('../lenna.png')
new_img = nearestInsertValue(img, 800, 800)
cv2.imshow('old img', img)
cv2.imshow('new img', new_img)
cv2.waitKey(0)