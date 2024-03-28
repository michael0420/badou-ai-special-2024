'''
    第二次作业：RGB图像的灰度化与二值化
'''

import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np

# 图像灰度化
img = cv2.imread("lenna.png")
h, w = img.shape[:2]
targetImage = np.zeros([h, w], img.dtype)

for high in range(h):
    for width in range(w):
        BGR = img[high, width]
        targetImage[high, width] = int(BGR[0] * 0.11 + BGR[1] * 0.59 + BGR[2] * 0.3)     # 遵循opencv读图通道顺序BGR

# 调用接口方法
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayPhoto = rgb2gray(img)

# cv2.imshow("origin_photo", img)
# cv2.imshow("gray_photo_1", targetImage)
# cv2.imshow("gray_photo_2", grayImage)
# cv2.imshow("gray_photo_3", grayPhoto)
# cv2.waitKey(0)

img = plt.imread('lenna.png')
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(grayImage, cmap='gray')
plt.subplot(223)
plt.imshow(grayPhoto, cmap='gray')

print(img)
print(targetImage)
print(grayImage)


# 图像二值化
high_1, width_1 = targetImage.shape
binaryImage = np.zeros([high_1, width_1], targetImage.dtype)
for i in range(high_1):
    for j in range(width_1):
        if (targetImage[i, j] / 255 >= 0.5):
            binaryImage[i, j] = 1
        else:
            binaryImage[i, j] = 0

# cv2.imshow("binaryImage", binaryImage)
# cv2.waitKey(0)

plt.subplot(224)
plt.imshow(binaryImage, cmap='gray')
plt.show()

print(binaryImage)
