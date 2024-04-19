import cv2
import numpy as np


def reshape_img(new_height, new_width, img):
    height, width = img.shape[:2]
    if height == new_height and width == new_width:
        return img
    scale_x = float(width) / new_width
    scale_y = float(height) / new_height
    new_img = np.zeros((new_height, new_width, 3), np.uint8)
    for r in range(new_height):
        for c in range(new_width):
            for i in range(3):
                src_x = (c + 0.5) * scale_x - 0.5
                src_y = (r + 0.5) * scale_y - 0.5
                x0 = int(np.floor(src_x))
                y0 = int(np.floor(src_y))
                x1 = min(x0 + 1, width - 1)
                y1 = min(y0 + 1, height - 1)
                temp0 = (src_x - x0) * img[y0, x1, i] + (x1 - src_x) * img[y0, x0, i]
                temp1 = (src_x - x0) * img[y1, x1, i] + (x1 - src_x) * img[y1, x0, i]
                new_img[r, c, i] = int((y1 - src_y) * temp0 + (src_y - y0) * temp1)

    return new_img


img = cv2.imread('img/lenna.png')
new_height = 700
new_width = 700
new_img = reshape_img(new_height, new_width, img)
cv2.imshow('Original Image', img)
cv2.imshow('Reshaped Image', new_img)
cv2.waitKey(0)
