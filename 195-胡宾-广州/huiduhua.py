from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

tupian = cv2.imread('lenna.png')
tupian = cv2.cvtColor(tupian, cv2.COLOR_BGR2RGB)
print(tupian)
# 灰度化
h, w = tupian.shape[:2]  # 获取图片的high和wide
tupian_gray = np.zeros([h, w], tupian.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        a = tupian[i, j]
        tupian_gray[i, j] = int(a[0] * 0.3 + a[1] * 0.59 + a[2] * 0.11)
cv2.imshow("image show gray", tupian_gray)
plt.subplot(223)
plt.imshow(tupian_gray, cmap='gray')
plt.show()
