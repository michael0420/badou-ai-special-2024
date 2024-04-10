

import numpy as np
import cv2
import matplotlib.pyplot as plt

'''彩图转换为 灰度图、二值化图'''

img = cv2.imread("../lenna.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]  # 获取图片的high和wide
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
img_binary = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的RGB坐标
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将RGB坐标转化为gray坐标并赋值给新图像
        img_binary[i, j] = img_gray[i, j]
        m = (img_binary[i, j]) / 255
        if(m < 0.5):
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 1

# 原图
plt.subplot(221)
print("image show BGR: %s" % img)
plt.imshow(img)

# 灰度图
plt.subplot(222)
print("image show gray: %s" % img_gray)
plt.imshow(img_gray, cmap='gray')

# 二值化图
plt.subplot(223)
print("image show binary: %s" % img_binary)
plt.imshow(img_binary, cmap='gray')

# 开启画图
plt.show()




