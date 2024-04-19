"""
@author: QiPiaoYang

最临近插值

"""

import cv2
import numpy as np


def function(srcImg, zoomRows, zoomCols):
    height, width, channels = srcImg.shape
    emptyImage = np.zeros((zoomRows, zoomCols, channels), np.uint8)
    sh = zoomRows / height
    sw = zoomCols / width
    for i in range(zoomRows):
        for j in range(zoomCols):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            emptyImage[i, j] = srcImg[x, y]
    return emptyImage


img = cv2.imread("../lenna.png")
zoom = function(img, 800, 800)
cv2.imshow("nearest", zoom)
cv2.imshow("srcImg", img)
cv2.waitKey(0)

dsize = (800, 800)
resized = cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)
cv2.imshow("cvNearest", resized)
cv2.waitKey(0)
