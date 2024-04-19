# -*- coding: utf-8 -*-
import numpy as np
import cv2

# 灰度化和二值化

# 图片读取
wtq_img0 = cv2.imread('../lenna.png')
# cv2 BGR转RBG
wtq_img = cv2.cvtColor(wtq_img0, cv2.COLOR_BGR2RGB)

h, w = wtq_img.shape[:2]
wtq_img_gray = np.zeros([h, w], wtq_img.dtype)
wtq_img_binarization = np.zeros([h, w], wtq_img.dtype)

# RBG转灰度
for i in range(h):
    for j in range(w):
        m = wtq_img[i, j]
        wtq_img_gray[i, j] =int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

# 灰度转二值
for i in range(h):
    for j in range(w):
        m = wtq_img_gray[i,j]
        if m / 255 <= 0.5:
            wtq_img_binarization[i, j] = 0
        else:
            wtq_img_binarization[i, j] = 255


cv2.imshow("lenna", wtq_img0)
cv2.imshow("gray", wtq_img_gray)
cv2.imshow("binarization", wtq_img_binarization)
cv2.waitKey(0)