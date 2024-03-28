import cv2
import numpy as np


def function(img, new_height, new_width):
    height, width, channels = img.shape
    emptyImage = np.zeros((new_height, new_width, channels), np.uint8)
    sh = new_height / height
    sw = new_width / width
    for i in range(new_height):
        for j in range(new_width):
            x = int(i / sh + 0.5)  # int(),转为整型，使用向下取整。
            y = int(j / sw + 0.5)
            # print("x=",x)
            emptyImage[i, j] = img[x, y]
    return emptyImage


# cv2.resize(img, (800,800,c),near/bin)

img = cv2.imread("lenna.png")
# 输入想要扩大的尺寸
zoom = function(img, 999, 999)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)


