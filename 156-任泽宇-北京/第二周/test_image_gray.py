from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("lenna.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#原图
plt.subplot(221)
plt.imshow(img)
# 灰度化
h,w = img.shape[:2]
# print(h,w)
#获取图片的high和wide
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

plt.subplot(222)
plt.imshow(img_gray,cmap='gray')

# 二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 113):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 255
plt.subplot(223)
plt.imshow(img_gray, cmap='gray')
plt.show()


