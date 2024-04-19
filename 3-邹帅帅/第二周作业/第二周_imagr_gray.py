# -- coding : utf-8 --
"""
 彩色图像的灰度化，二值化
"""
# skimage.color 是 Scikit-image 库中的一个子模块,包含了多种颜色空间之间的转换函数,rgb2gray：将RGB图像转换为灰度图像
from skimage.color import rgb2gray

# numpy 提供了多维数组和矩阵运算能力
import numpy as np

# pyplot子库提供了大量的图形展示功能,plt.imshow()来显示图像，plt.subplot 图像窗口（Figure）中创建和排列多个子图（Axes）(221)2行2列第一个图
import matplotlib.pyplot as plt

# PIL图像处理库，它可以打开、操作和保存许多不同格式的图像文件。Image模块，可以方便地进行图像的读取、显示、转换和保存等操作。
from PIL import Image

# 计算机视觉库，提供了大量的图像和视频处理功能。加载/保存图像、图像变换、图像过滤、特征检测、物体识别等等。
import cv2

# cv2.imread 是 OpenCV 库中的一个函数，用于读取图像文件并将图像数据加载到内存中作为一个 NumPy 数组。
img = cv2.imread('lenna.png')

# NumPy中，一个数组（或矩阵）的shape属性是一个表示数组各维度长度的元组
# 图像数据，(height, width, 3) 表示图像的高度、宽度及3个颜色通道（RGB）
h,w = img.shape[:2]  #取图片高宽

img_gray = np.zeros([h,w],img.dtype)  #创建一个和原图片高、宽、类型相同的单通道图片

for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)


print(m)
print(img_gray)
print("image show gray: %s"%img_gray) # %s格式化 %后面的量 使其变为字符串类型
print(type(img_gray))
cv2.imshow("img show gray",img_gray) # 用于显示图像，第一个变量字符串，显示图像窗口名称；第二个变量，是要显示的图像，通常为数组
"""
cv2.waitKey(0) # 0 表示无限期等待，直到用户按键才继续
在调用 cv2.imshow() 之后，必须紧跟 cv2.waitKey() 函数，它会暂停程序执行并等待用户按键，同时保持图像窗口打开。
如果不这样做，窗口可能会立即关闭导致你看不到图像
"""

plt.subplot(221)
img = plt.imread("lenna.png") # 读取图片数组 RGB模式
# img = cv2.imread("lenna.png", False) # 1：彩图 0（False）：灰度图 -1：包括图像的 alpha 通道（如果存在）
plt.imshow(img)
print("---image lenna---")
print(img)

# 灰度化
img_gray = rgb2gray(img)  #rgb2gray：将RGB图像转换为灰度图像
# img = cv2.imread("lenna.png")
# img_gray = cv2.cvtColor(img,COLOR_BGR2GRAY) 转换为灰度图像
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
print("---image gray---")
print(img_gray)

# 二值化
#img_binary = img_gray / 255
#rows,cols = img_grat.shape
#for i in range(rows):
#    for j in range(cols):
#        if(img_binary[i,j] < 0.5)
#            img_binary[i,j] = 0
#        else:
#            img_binary[i,j] = 1

img_binary = np.where(img_gray >= 0.5,1,0)
print("---img binary---")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()







