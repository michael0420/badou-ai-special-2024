import numpy as np
import matplotlib as plt
import cv2

image = cv2.imread("../lenna.png")
src_h, src_w, channel = image.shape
dst_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
scale_x, scale_y = float(src_w) / 1000, float(src_h) / 1000
for i in range(channel):
    for dst_y in range(1000):
        for dst_x in range(1000):
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5

            src_x0 = int(np.floor(src_x))
            src_y0 = int(np.floor(src_y))

            # src_w - 1 为边界位置
            src_x1 = min(src_x0 + 1 ,src_w - 1)
            src_y1 = min(src_y0 + 1, src_h - 1)

            temp0 = (src_x1 - src_x) * image[src_y0, src_x0, i] + (src_x - src_x0) * image[src_y0, src_x1, i]
            temp1 = (src_x1 - src_x) * image[src_y1, src_x0, i] + (src_x - src_x0) * image[src_y1, src_x1, i]
            dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

cv2.imshow('image', image)
cv2.imshow('bilinear interp',dst_img)
cv2.waitKey()


