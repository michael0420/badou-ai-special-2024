# _*_ coding: UTF-8 _*_
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray

# 灰度化方法一
image = cv2.imread('../data/lenna.png')
h, w = image.shape[0], image.shape[1]
# 生成相同尺寸的数组
gray_img = np.zeros([h, w], dtype=image.dtype)
for i in range(h):
    for j in range(w):
        data = image[i, j]
        # 由于opencv读取的数据为BGR
        gray_img[i, j] = int(0.3 * data[2] + 0.59 * data[1] + 0.11 * data[0])

# 显示原图
image = plt.imread('../data/lenna.png')
plt.subplot(3, 2, 1)
plt.imshow(image)

# 显示灰度图
plt.subplot(3, 2, 2)
plt.imshow(gray_img, cmap='gray')
# 方法二
image = Image.open('../data/lenna.png')
convert_img = image.convert('L')
# convert_img.show()
plt.subplot(3, 2, 3)
plt.imshow(convert_img, cmap='gray')

# 方法三
image = cv2.imread('../data/lenna.png')
cvt_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(3, 2, 4)
plt.imshow(cvt_img, cmap='gray')

# 方法四
image = cv2.imread('../data/lenna.png')
gray_img = rgb2gray(image)
plt.subplot(3, 2, 5)
plt.imshow(gray_img, cmap='gray')

plt.show()
