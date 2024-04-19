from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img1 = cv2.imread('lenna.png')
high,wide = img1.shape[:2]
img_gray = np.zeros([high,wide],img1.dtype)
for i in range(high):
    for j in range(wide):
        m = img1[i,j]
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

#二值化
img_gray = rgb2gray(img)
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')

plt.show()

