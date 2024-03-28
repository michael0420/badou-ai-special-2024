import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('img.jpg')  # 默认使用cv2.IMREAD_COLOR模式，读出来是BGR
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)  # 2维数组，用于存灰度图

for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.114 + m[1] * 0.587 + m[2] * 0.299)  # 灰度算法：

# cv2.imshow("image show bgr", img)  # 用cv，可以直接显示BGR图片
# cv2.imshow("image show gray", img_gray)  # 用cv，可以直接显示BGR图片
print('-----------------img bgr---------------------')
print(img)  # BGR图片是三维数组
print('-----------------img gray---------------------')
print(img_gray)  # 灰度图片是二维数组
cv2.waitKey(0)  # 等待用户操作，防止窗口一闪而过

plt.subplot(221)  # plt.subplot函数是用来创建一个子图网格的，它允许你在单个图形窗口中展示多个图表，2行2列的网格，当前激活第一个子图位置（左上角）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv默认是BGR，需转成RGB才能在plt中正确展示
plt.imshow(img_rgb)
print('-----------------img rgb---------------------')
print(img_rgb)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')  # cmap表示颜色映射的方式，gray是灰度

# 二值化
rows, cols = img_gray.shape
img_binary = np.zeros([h, w], img.dtype)
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] < 128:
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 1

# img_gray是二维数组，在numpy中可以直接和数组进行比较，是因为numpy有广播机制，可自动将较小的维度的数据扩展到较大维度的数据的形状，用以匹配维度
# img_binary = np.where(img_gray <= 0.5, 1, 0)
print("-----img_binary------")
print(img_binary)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
