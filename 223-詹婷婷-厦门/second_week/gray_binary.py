import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray


img = plt.imread("lenna.png")
print(img)
plt.subplot(321)
plt.imshow(img, cmap='gray')

img1 = cv2.imread("lenna.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.subplot(322)
plt.imshow(img1, cmap='gray')

h, w = img.shape[:2]
img_gray = np.zeros([h,w], img.dtype)
for i in range(h):
    for j in range(w):
        img_gray[i,j] = img[i,j][0] * 0.3 + img[i,j][1] * 0.59 + img[i,j][2] * 0.11
print(img_gray)
plt.subplot(323)
plt.imshow(img_gray, cmap='gray')

img_gray1 = rgb2gray(img)
print(img_gray1)
plt.subplot(324)
plt.imshow(img_gray1, cmap='gray')

img_binary = np.zeros([h,w], img.dtype)
sum = 0
for i in range(h):
    for j in range(w):
        sum += img_gray[i,j]
avg = sum / (h * w)
print(sum)
for i in range(h):
    for j in range(w):
        if img_gray[i,j] > avg:
            img_binary[i,j] = 1
        else:
            img_binary[i,j] = 0
print(img_binary)
plt.subplot(325)
plt.imshow(img_binary, cmap='gray')




plt.show()
