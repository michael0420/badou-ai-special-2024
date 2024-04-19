# 单行注释
"""
彩色图像二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

# 二值化原始方法

# 获取文件路径
two_levels_up = os.path.join(os.getcwd(), os.path.pardir, os.path.pardir)
images_dir = os.path.join(two_levels_up, "resources")
image_path = os.path.join(images_dir, "lenna.png")
img = cv2.imread(image_path)

# 获取图片的高宽
h, w = img.shape[:2]
# 创建单通道图片
image_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        # 取出当前high和wide中的BGR坐标
        m = img[i, j]
        # 将BGR坐标转化为gray坐标并赋值给新图像
        image_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print(m)
print(image_gray)
print("image show gray: %s" % image_gray)
cv2.imshow("image show gray", image_gray)

plt.subplot(221)
img = plt.imread(image_path)
# img = cv2.imread(image_path, False)
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

# 原始二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
print("-------二值化图片-------")
# img_binary = np.where(img_gray >= 0.5, 1, 0)
print(img_gray)
print(img_gray.shape)

plt.subplot(223)
plt.imshow(img_gray, cmap='gray')
plt.show()

