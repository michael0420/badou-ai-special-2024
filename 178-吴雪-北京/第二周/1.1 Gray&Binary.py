import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

img = cv2.imread("E:/Desktop/jianli/lenna.png")
h, w = img.shape[:2]  # 获取图片的high和wide
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]  # 取出图片img当前的high和wide中的BGR坐标
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
print(img_gray)
print("image show gray:%s"%img_gray)
cv2.imshow("image show gray", img_gray)
# cv2.waitKey(0)  # 保持窗口显示
img = plt.imread("E:/Desktop/jianli/lenna.png")  # plt在matplotbil库里，使用RGB格式
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title("image")  # 创建图像窗口
print("----image lenna----")
print(img)
# 灰度化  两种方法
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(222), plt.imshow(img_gray, cmap='gray'), plt.title("gray")
print("----image gray----")
print(img_gray)
# 二值化
rows,cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
img_binary = img_gray
# img_binary = np.where(img_gray >= 0.5, 1, 0)
print("----image binary----")
print(img_binary)
print(img_binary.shape)
plt.subplot(223), plt.imshow(img_binary, cmap='gray'), plt.title("binary")
plt.show()
