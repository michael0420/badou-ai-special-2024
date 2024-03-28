from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 实现图片灰度化
img = cv2.imread("lenna.png")
img_gray = rgb2gray(img)
plt.subplot(221)
plt.imshow(img_gray, cmap='gray')

# 实现图片二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(222)
plt.imshow(img_binary, cmap='gray')

plt.show()
