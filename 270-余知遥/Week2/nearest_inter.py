import cv2
import numpy as np


def nearest_inter(img):
    height, width, channels = img.shape
    output_img = np.zeros((800, 800, channels), np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh +0.5)
            y = int(j/sw +0.5)
            output_img[i, j] = img[x, y]
    return output_img

img = cv2.imread('./lenna.png')
output_img = nearest_inter(img)
cv2.imshow('nearest_inter', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()