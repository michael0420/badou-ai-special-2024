import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

# 灰度化:方式一
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
cv2.imwrite("lenna_grade.png",img_gray)

# 灰度化：方式二
img1 = plt.imread("lenna.png")
plt.subplot(221)
plt.imshow(img1,cmap='Accent')
img2 = plt.imread("lenna.png")
img2_gray = rgb2gray(img2)
img3_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
plt.subplot(222)
plt.imshow(img2_gray,cmap='gray')
plt.subplot(223)
plt.imshow(img3_gray,cmap='gray')

# 二值化:方式一
rows ,cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i,j] / 255 <= 0.5):
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 1


# 二值化：方式二
img_binary = np.where(img2_gray >= 0.5,255,0)

plt.subplot(224)
plt.imshow(img_binary,cmap='gray')
cv2.imwrite("lenna_binary.png",img_binary)
plt.show()