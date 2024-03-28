import cv2
import numpy as np

image = cv2.imread('lenna.png')

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = image.shape[0:2]
iamgeBinary = np.zeros((h, w), imageGray.dtype)

for x in range(h):
    for y in range(w):
        if imageGray[x, y] / 255 < 0.5:
            iamgeBinary[x, y] = 0
        else:
            iamgeBinary[x, y] = 255

cv2.imshow("img", image)
cv2.imshow("img_binary", iamgeBinary)
cv2.waitKey()  # 等待用户按键输入，无限等待
cv2.destroyAllWindows()  # 等待用户按键输入，无限等待 释放资源
