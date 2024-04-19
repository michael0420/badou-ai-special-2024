"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/3/30 21:47
"""
import numpy as np
import cv2

# 读取图片
img = cv2.imread("lenna.png")
print(f"原图矩阵{img}")
cv2.imshow("img", img)

# 灰度化
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"灰度图矩阵{gray_img}")
cv2.imshow("gray_img", gray_img)

# 二值图
# 将图片量化转化为[0-1]区间的浮点数
binary_img = np.asarray(gray_img) / 255.0
print(binary_img)
w, h = gray_img.shape
for i in range(w):
    for j in range(h):
        if binary_img[i, j] <= 0.5:
            binary_img[i, j] = 0
        else:
            binary_img[i, j] = 1
print(f"二值图矩阵{binary_img}")
cv2.imshow("binary_img", binary_img)

# 邻近插值
# 新建一个空白的800x800图像
width, height, channels = img.shape
new_img = np.zeros((800, 800, channels), np.uint8)
# 计算新图与原图比例
width_bili = 800/width
height_bili = 800/height
for i in range(800):
    for j in range(800):
        # 计算当前像素与原像素最邻近的像素点
        x = round(i/width_bili)
        y = round(j/height_bili)
        # 将原图像素赋值给新图像素
        new_img[i, j] = img[x, y]
print(f"邻近插值矩阵{new_img}")
cv2.imshow("linjin_img", new_img)
cv2.waitKey()
