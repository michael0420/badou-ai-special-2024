import cv2
import numpy as np

original_img = cv2.imread('img.jpg')

# cv2.imshow('origin img', origin_img)
# cv2.waitKey(0)

scale_factor = 2  # 缩放系数，如：放大2倍

h, w, c = original_img.shape
new_h, new_w = int(h * scale_factor), int(w * scale_factor)

new_img = np.zeros([new_h, new_w, c], dtype=original_img.dtype)

for y in range(new_h):
    for x in range(new_w):
        original_x = x / scale_factor
        original_y = y / scale_factor
        nearest_y = min(int(original_y + 0.5), h - 1)
        nearest_x = min(int(original_x + 0.5), w - 1)
        new_img[y, x] = original_img[nearest_y, nearest_x]

cv2.imshow('new img', new_img)
cv2.waitKey(0)
