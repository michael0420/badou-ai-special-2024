import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread('lenna.png')


img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# 灰度
m, n = img.shape[:2]
img_gray = np.zeros([m, n], img.dtype)
for i in range(m):
    for j in range(n):
        data = img[i, j]
        img_gray[i, j] = int(data[0] * 0.3 + data[1] * 0.59 + data[2] * 0.11)
print(m)
print(n)

cv2.imshow("img", img_bgr)
cv2.imshow("image show gray", img_gray)

plt.subplot(221)
plt.imshow(img_gray, cmap='gray')



# 二值化
img_binary = np.zeros([m, n], img.dtype)
for i in range(m):
    for j in range(n):
        data = img_gray[i, j]
        if data > 100:
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 100
# print(img_gray)
# print(img_binary)
plt.subplot(222)
plt.imshow(img_binary, cmap='gray')

cv2.waitKey(0)
plt.show()
