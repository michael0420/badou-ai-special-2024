
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import matplotlib
from PIL import Image
import cv2

# 灰度化原理

img = cv2.imread("lenna.png")

print(type(img))
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
print(height)
print(width)
print(type(height))
print(type(width))
print("高: %s ,宽: %s ,通道：%s"%(height, width, channels))
gray_img = np.zeros([height, width], dtype=np.uint8)
# print(img.dtype)
# cv2.imshow("gray",gray_img)
# cv2.waitKey(0)
for i in range(height):
    for j in range(width):
        m = img[i, j]
        gray_img[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
print(gray_img)
cv2.imshow("gray image", gray_img)
# cv2.waitKey(0)



# 灰度化调库方法

plt.subplot(331)
img = plt.imread('lenna.png')
plt.imshow(img)

plt.subplot(332)
gray_img2 = rgb2gray(img)
plt.imshow(gray_img2, cmap='gray')
print(gray_img2)

plt.subplot(333)
gray_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img3, cmap='gray')


# 二值化原理
plt.subplot(334)
height2 = gray_img2.shape[0]
width2 = gray_img2.shape[1]

for i in range(height2):
    for j in range(width2):
        if (gray_img2[i, j] <= 0.5):
            gray_img2[i, j] = 0
        else:
            gray_img2[i, j] = 1
plt.imshow(gray_img2, cmap="gray")

# 二值化调库
# plt.subplot(335)
# img_binary = np.where(gray_img2 >= 0.5, 1, 0)
# plt.imshow(img_binary, cmap="gray")

plt.show()