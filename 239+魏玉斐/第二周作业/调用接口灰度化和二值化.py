import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
plt.subplot(221)
img = cv2.imread('lenna.png')
# 将BGR图像转换为RGB图像
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 原图
plt.imshow(img)
plt.title('original')
print('-------lenna.png--------')
# print(img)
plt.subplot(222)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_grey, cmap='gray')
print(img_grey)
print('-------grey--------')
plt.title('grey')
# 二值化
plt.subplot(223)
# 阈值处理
img_binary = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY)[1]
plt.imshow(img_binary, cmap='gray')
print('-------binary--------')
plt.title('binary')
plt.show()
