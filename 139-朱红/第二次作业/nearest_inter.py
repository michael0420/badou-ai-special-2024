"""
@author: zhuhong

最临近插值
"""

import cv2
import numpy as np

def function(img, height, width):
    h, w, c = img.shape
    emptyImage = np.zeros((height, width, c), np.uint8)
    sh = height / h
    sw = width / w
    for i in range(height):
        for j in range(width):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage

img = cv2.imread("lenna.png")
zoom = function(img, 800, 900)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("original image", img)
cv2.waitKey(0)