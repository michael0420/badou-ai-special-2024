import cv2
import numpy as np

def function(img):
    h, w, c = img.shape
    emptyImage = np.zeros((800, 800, c), img.dtype)
    sh = 800 / h
    sw = 800 / w
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)  # int转为整形 向下取整
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage

img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("img", img)
cv2.waitKey(0)

