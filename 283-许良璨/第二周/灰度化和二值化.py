import cv2
import numpy as np
#灰度化
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray=np.zeros([h,w],img.dtype)
for i in range(h):
    for g in range(w):
        m=img[i,g]
        img_gray[i,g]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)

print(img_gray)
cv2.imshow("完成",img_gray)

#二值化
img_binary=np.zeros([h,w],img.dtype)
h1,w1 = img_gray.shape
for i in range(h1):
    for g in range(w1):
        if img_gray[i,g]<0.5:
            img_binary[i,g]=0
        else:
            img_binary[i, g] = 1

cv2.imshow("完成",img_binary)
