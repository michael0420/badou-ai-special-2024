import cv2
import numpy as np

# 图片最邻近插值
def nearest_interpolation(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800 / height
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread("lenna.png")
zoom = nearest_interpolation(img)
cv2.imshow("interpolation img", zoom)
cv2.imshow("original img ", img)
cv2.waitKey(0)
