
"""
实现图像最临近插值
"""

import cv2
import numpy as np

def nearInsert(img,h,w):
    high,wide,channels = img.shape
    emptyImg = np.zeros([h,w,channels],np.uint8)
    sh = h/high
    sw = w/wide
    for i in range(h):
        for j in range(w):
            x = int(i/sh + 0.5)     #转为整型，向下取整
            y = int(j/sw + 0.5)
            emptyImg[i,j] = img[x,y]
    return emptyImg


img = cv2.imread("lenna.png")
res_img = nearInsert(img,800,800)
cv2.imshow("nearest interp",res_img)
cv2.waitKey(0)
