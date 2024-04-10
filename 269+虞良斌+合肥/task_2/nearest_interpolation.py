import numpy as np
import cv2
import matplotlib as plt

'''最邻近插值实现图片的缩放'''

img = cv2.imread("../lenna.png")

h, w, c = img.shape
empty_image = np.zeros((1000, 1000, c), np.uint8)
h_ratio = 1000 / h
w_ratio = 1000 / w
for i in range(1000):
    for j in range(1000):
        h_src = round(i / h_ratio)
        w_src = round(j / w_ratio)

        # 防止因四舍五入导致的数组越界
        if (h_src >= img.shape[0]):
            h_src = h_src - 1
        if (w_src >= img.shape[1]):
            w_src = w_src - 1
        empty_image[i, j] = img[h_src, w_src]

cv2.imshow("image", img)
cv2.imshow("nearest image", empty_image)
cv2.waitKey(0)

