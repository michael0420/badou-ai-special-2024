import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png', 1)
print(img)
rows, columns = img.shape[:2]
img_gray = np.zeros([rows, columns], img.dtype)

for j in range(columns):
    for i in range(rows):
        img_gray[i, j] = int(img[i, j, 0] * 0.11 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.3 + 0.5)


plt.subplot(222)
dst2 = plt.imshow(img_gray, cmap=plt.cm.gray)
plt.colorbar(dst2, cax=None, ax=None)

plt.subplot(221)
img_color = plt.imread("lenna.png")
dst1 = plt.imshow(img_color)
print(img_color)
plt.colorbar(dst1, cax=None, ax=None)

rows, columns = img_gray.shape[:2]
img_binary = np.zeros([rows, columns], img_gray.dtype)
for j in range(columns):
    for i in range(rows):
        if img_gray[i, j] <= 0.5*256:
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 255
plt.subplot(223)
dst3 = plt.imshow(img_binary, cmap=plt.cm.gray)
plt.colorbar(dst3, cax=None, ax=None)
plt.show()
