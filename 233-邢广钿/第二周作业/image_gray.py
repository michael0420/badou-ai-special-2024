import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt


# ===将彩色图像灰度化(原生写法)=====
# 读取图片
img = cv2.imread("lenna.png")
# 获取图片的高度、宽度
high, wide = img.shape[:2]
# 新建一张空白图
img_gray = np.zeros([high, wide], img.dtype)
# 遍历原图
for i in range(high):
    for j in range(wide):
        # 获取原图像素点
        m = img[i, j]
        # 将原图像素点通过公式计算赋值到空白图
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
# 显示图片
cv2.imshow("image", img_gray)
# 等待按键命令后才关闭图片窗口
key = cv2.waitKey(0)


# ===将彩色图像灰度化(调用CV写法)=====

img = plt.imread("lenna.png")
img_gray1 = rgb2gray(img)  # 调用函数转换（方式1）
img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)  # 调用函数转换（方式2）
# 2*3图片模板 第1个位置
plt.subplot(231)
plt.imshow(img)

plt.subplot(232)
plt.imshow(img_gray1, cmap='gray')

plt.subplot(233)
plt.imshow(img_gray2, cmap='gray')

# ====二值化=======
# 写法1
rows, cols = img_gray1.shape
for i in range(rows):
    for j in range(cols):
        img_gray1[i, j] = 0 if img_gray1[i, j] <= 0.5 else 1
# 写法2
img_gray2 = np.where(img_gray2 > 0.5, 1, 0)
plt.subplot(234)
plt.imshow(img_gray1, cmap='gray')

plt.subplot(235)
plt.imshow(img_gray2, cmap='gray')
plt.show()



