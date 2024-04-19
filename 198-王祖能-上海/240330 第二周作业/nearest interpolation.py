import numpy as np
import matplotlib.pyplot as plt
import cv2

def function(img, x, y):
    h, w, c = img.shape
    size_1, size_2 = x / h, y / w  # size为尺寸放大比例系数
    empty = np.zeros([x, y, c], np.uint8)
    # print(empty)
    for i in range(x):
        for j in range(y):
            m = int(i / size_1 + 0.5)
            n = int(j / size_2 + 0.5)
            empty[i, j] = img[m, n]
    print(empty)
    cv2.imshow('demo1', img)
    cv2.imshow('demo2', empty)
    cv2.waitKey(0)
img = cv2.imread('lenna.png')
function(img, 200, 200)





