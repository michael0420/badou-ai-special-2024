"""调用接口灰度化&二值化
invocation interface gray&binary"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
# img = plt.imread("E:/Desktop/jianli/lenna.png")
img = cv2.imread("E:/Desktop/jianli/lenna.png")
img_bgr2rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_binary = np.where((img_gray/255.0) >= 0.5, 1, 0)
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(221), plt.imshow(img_bgr2rgb, cmap='gray'), plt.title("image")
plt.subplot(222), plt.imshow(img_gray, cmap='gray'), plt.title("gray")
plt.subplot(223), plt.imshow(img_binary, cmap='gray'), plt.title("binary")
plt.show()
