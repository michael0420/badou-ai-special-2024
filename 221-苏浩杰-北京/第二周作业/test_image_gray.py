"""
彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

imread = cv2.imread("lenna.png")
h,w = imread.shape[:2]
image_gray = np.zeros([h, w], imread.dtype)
for i in range(h):
    for j in range(w):
        m=imread[i,j]
        aa = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
        image_gray[i,j]=aa
print("cv gray image show : %s"%image_gray)
cv2.imshow("image gray",image_gray)
# cv2.waitKey(0)

img=plt.imread("lenna.png")
plt.subplot(221)
plt.imshow(img)


plt.subplot(222)
img=plt.imread("lenna.png")
img_gray = rgb2gray(img)
print("plt gray image show:%s"%img_gray)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.imshow(img_gray, cmap='gray')

second_binary_image = np.where(img_gray > 0.5, 1, 0)
print("second  image show:%s"%second_binary_image)
plt.subplot(223)
plt.imshow(second_binary_image,cmap="gray")

plt.show()