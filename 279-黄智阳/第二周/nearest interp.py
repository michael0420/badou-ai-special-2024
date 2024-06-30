import numpy as np
import cv2

def function(img):
    height, width, channels = img.shape
    enptyImage = np.zeros((800,800,channels),np.uint8)  # 800,800是新图后的长宽
    sh = 800/height
    sw = 800/width  # 放大的倍数
    for i in range(800):
        for j in range(800):
            x = int(i/sh+0.5)
            y = int(j/sw+0.5)
            enptyImage[i,j] = img[x,y]
    return enptyImage

# 最临近插值
img = cv2.imread("lenna.png")
zoom = function(img)
cv2.imshow("win_img",img)
cv2.imshow("win_big",zoom)
cv2.waitKey(0)
