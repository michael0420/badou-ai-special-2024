import cv2
import numpy as np


def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((1000, 1000, channels), np.uint8)
    sh = 1000 / height
    sw = 1000 / width
    for i in range(1000):
        for j in range(1000):
            x = int(i / sh + 0.5)  # int(),转为整型，使用向下取整。
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


# cv2.resize(img, (800,800,c),near/bin)

img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)