"""
练习图片灰度化处理
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# 1.图片灰度化处理 ，方法1
img = cv2.imread("lenna.png")  # 读取图片文件
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 将BGR通道，转换为RGB通道
# print(img.shape)  # (512, 512, 3) 图片读取像素行数512，列数512，3通道
h, w = img.shape[0:2]  # h 和 w分别为img的行数和列数
gray_img = np.zeros((h, w), img.dtype)   # 创建一张与原图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]
        gray_img[i, j] = int(m[0]*0.3 + m[1]*0.59 + m[2]*0.11)  # 将原图片坐标为i，j的像素点灰度处理后赋值给单通道图片坐标i，j的位置上

# print(m)
# print(gray_img)
cv2.imshow("img", gray_img)
# key=cv2.waitKey()  # 让imag窗口持续显示图片

# 创建一个创建，并将原始图片放在窗口的第一个位置
plt.subplot(221)  # 将整个窗口分为2行2列，当前为第一个位置
img = plt.imread("lenna.png")
plt.imshow(img)

# print("image lenna")
# print(img)

# 图片灰度化处理，方式2：
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap="gray")


# 二值化，方式1：
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if img_gray[i, j] < 0.5:
#             img_gray[i, j] = 0  # 该点像素置黑
#         else:
#             img_gray[i, j] = 1  # 该点像素置白

# 二值化，方式2：
img_binary = np.where(img_gray < 0.5, 0, 1)
plt.subplot(223)
plt.imshow(img_binary, cmap="gray")
plt.show()
