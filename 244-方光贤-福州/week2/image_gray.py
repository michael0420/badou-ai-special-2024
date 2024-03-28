import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray

img = cv2.imread("lenna.png")   #读图 cv模式
plt.subplot(221)
plt.imshow(img)
height, width = img.shape[:2]  #读高度和宽度，3为channel
#灰度化 cv模式

gray_img = np.zeros([height,width], img.dtype)  #创建等大小零数组
#创建一个for循环 遍历矩阵中所有rgb的坐标
for i in range(height):
    for j in range(width):
        k = img[i, j]  #读取坐标
        gray_img[i, j] = int(k[0] * 0.11 + k[1] * 0.59 + k[2] * 0.3)  #转灰度公式

cv2.imshow("gray", gray_img)
cv2.waitKey(0)

img = plt.imread("lenna.png")   #读图 plt模式
plt.imshow(img)

plt.subplot(222)
#快捷灰度化
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 法1
gray_img = rgb2gray(img) #法2
plt.imshow(gray_img, cmap='gray')

#二值化 cv模式

binary_img = np.ones([height,width],img.dtype)
#创建一个for循环 遍历矩阵中所有rgb的坐标 并作比较
for i in range(height):
    for j in range(width):
        if (gray_img[i, j] <= 0.5) :
            binary_img[i, j] = 0
        else:
            binary_img[i, j] = 1
cv2.imshow("binary", binary_img)
cv2.waitKey(0)

#快捷二值化
binary_img1 = np.where(gray_img >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(binary_img1, cmap='gray')

binary_img1 = np.where(gray_img <= 0.5, 1, 0)
plt.subplot(224)
plt.imshow(binary_img1, cmap='gray')

plt.show()
