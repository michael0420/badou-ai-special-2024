"""
实现灰度化和二值化
@author: zsj
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

original_img = cv2.imread('../lenna.png')  # 以BGR读取
print('original img', original_img)
cv2.imshow("original img", original_img)  # cv2本身的展示是正确的


# 实现灰度化
def imgToGray(img):
    height, width = img.shape[:2]
    # 初始化空图片
    img_gray = np.zeros([height, width], img.dtype)
    for i in range(height):
        for j in range(width):
            xs = img[i, j]
            # img_gray[i, j] = int((xs[0]*30 + xs[1]*59 + xs[2]*11) / 100)  顺序错误
            img_gray[i, j] = int((xs[0] * 11 + xs[1] * 59 + xs[2] * 30) / 100)
            # img_gray[i, j] = int(xs[0] * 0.11 + xs[1] * 0.59 + xs[2] * 0.3)
    return img_gray


plt.subplot(221)
# img_rgb = plt.imread('lenna.png')
img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转为 RGB
print('rgb img', img_rgb)
plt.imshow(img_rgb)

plt.subplot(222)
new_img = imgToGray(original_img)
print('new img', new_img)
plt.imshow(new_img, cmap='gray')


# 实现二值化
def imgGrayToBinary(img_gray):
    height, width = img_gray.shape[:2]
    # 初始化空图片
    img_binary = np.zeros([height, width], img_gray.dtype)
    mid = 255 / 2
    for i in range(height):
        for j in range(width):
            if img_gray[i, j] > mid:
                img_binary[i][j] = 255
            else:
                img_binary[i][j] = 0
    return img_binary


plt.subplot(223)
img_bin = imgGrayToBinary(new_img)
print('binary img', img_bin)
plt.imshow(img_bin, cmap='gray')

plt.show()
