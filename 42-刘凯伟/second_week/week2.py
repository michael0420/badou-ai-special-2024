import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray

img1 = cv2.imread("lenna.png")

# 灰度化
h,w = img1.shape[:2]
img_gray = np.zeros([h,w],img1.dtype)
for i in range(h):
    for j in range(w):
        m = img1[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
cv2.imshow("image show gray",img_gray)
cv2.waitKey(0)


img = plt.imread("lenna.png")
plt.subplot(221)
plt.imshow(img)

# 灰度化
img_gray1 = rgb2gray(img)

# 二值化
# 方法一
img_binary = np.where(img_gray1 >= 0.5, 1, 0)
plt.subplot(222)
plt.imshow(img_binary, cmap='gray')

# 方法二
rows, cols = img_gray1.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray1[i, j] <= 0.5):
            img_gray1[i, j] = 0
        else:
            img_gray1[i, j] = 1

plt.subplot(223)
plt.imshow(img_gray1, cmap='gray')


# 最邻近插值
height,width,channels =img1.shape
emptyImage=np.zeros((1000,1000,channels),np.uint8)
sh=1000/height
sw=1000/width
for i in range(1000):
    for j in range(1000):
        x = int(i / sh + 0.5)
        y = int(j / sw + 0.5)
        emptyImage[i, j] = img1[x, y]
plt.subplot(224)
plt.imshow(emptyImage)


plt.show()