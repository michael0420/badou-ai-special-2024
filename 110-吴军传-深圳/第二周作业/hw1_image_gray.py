from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 恢复值
img = cv2.imread("lenna.png")  # 插入图片
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前的h和w的BGR坐标
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
print(m)
print(img_gray)
# cv2.imshow("lenna gray", img_gray)
img_gray = rgb2gray(img)  #将彩色图像转换为灰度图像
plt.subplot(221)
plt.imshow(img_gray, cmap="gray")
print("---image gray----")
print(img_gray)

# 二值化
rows, cols = img_gray.shape  #
for r in range(rows):
    for c in range(cols):
        if img_gray[r, c]  <= 0.5:
            img_gray[r, c] = 0
        else:
            img_gray[r, c] = 1
print("-----imge_binary------")
print(img_gray)
print(img_gray.shape)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
plt.show()
