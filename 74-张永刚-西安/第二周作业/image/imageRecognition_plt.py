"""
@author: 74+张永刚
彩色图像的二值化
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 创建一个2*2的画布，并将此次图片放在画布第1个位置
plt.subplot(221)
# 读取图片，读取到的内容是一个矩阵
img = plt.imread("img/lenna.png")
# 这里没有做任何处理，打印出来是原图
# 将要显示的图片存起来
plt.imshow(img)

# 灰度化
# 创建一个2*2的画布，并将此次图片放在画布第2个位置
plt.subplot(222)
img_gray = rgb2gray(img)
# 将要显示的图片存起来
plt.imshow(img_gray,cmap='gray')


# 二值化1
rows,cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        # print(img_gray[i,j]) 这里打印出来看一下输出什么内容
        if img_gray[i,j] <= 0.5:
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 1
# 创建一个2*2的画布，并将此次图片放在画布第3个位置
plt.subplot(223)
# 将要显示的照片存起来
plt.imshow(img_gray,cmap='gray')


# 二值化2
img_gray2 = rgb2gray(img)
img_gray2 = np.where(img_gray2 <= 0.5 ,0 ,1)
# 创建一个2*2的画布，并将此次图片放在画布第4个位置
plt.subplot(224)
plt.imshow(img_gray2,cmap='gray')

# 最终显示画布所有内容
plt.show()




