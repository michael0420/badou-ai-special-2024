"""
@author 193-杨婷
彩色图像二值化处理
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
plt.rcParams['font.family'] = 'SimHei'  # 设置中文字体为黑体
"""
matplotlib 读取的图像数据范围是 0 到 1 之间的浮点数，表示像素值在 0 到 255 之间的归一化显示
而 OpenCV 读取的图像数据是原始像素值（在 0 到 255 之间的整数）>> 归一化：x/255
"""
img = plt.imread("lenna.png")
# print(img)
img_gray = rgb2gray(img)
# print("--------------------------")
# print(img_gray)

# 手撸二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
plt.subplot(121)
plt.title("手撸结果展示")
plt.imshow(img_gray, cmap="gray")

# 调用函数二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----image_binary------")
print(img_binary)
print("****************************")
print(img_binary.shape)
plt.subplot(122)
plt.title("函数结果展示")
plt.imshow(img_binary, cmap="gray")
plt.show()



