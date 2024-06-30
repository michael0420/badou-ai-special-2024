import cv2
import numpy as np


def bgr2gray(img_bgr):
    height, width, channels = img_bgr.shape
    img_gray = np.zeros([height, width, 1], img_bgr.dtype)
    for x in range(width):
        for y in range(height):
            m = img_bgr[x,y]
            img_gray[x,y] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
    return img_gray

def gray2binary(img_gray):
    height, width, channels = img_gray.shape
    img_binary = np.zeros([height, width, 1], img_gray.dtype)
    for x in range(width):
        for y in range(height):
            if (img_gray[x,y] <= 127):
                img_binary[x,y] = 0
            else:
                img_binary[x,y] = 255
    return img_binary


img = cv2.imread('./lenna.png')
img_gray = bgr2gray(img)
cv2.imshow('img_gray', img_gray)
img_binary = gray2binary(img_gray)
cv2.imshow('img_binary', img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()




