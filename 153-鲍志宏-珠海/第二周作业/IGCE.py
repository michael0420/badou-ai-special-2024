
#图片灰度化练习


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray



img=cv2.imread("lenna.png")
#读取图片
h,w=img.shape[:2]
#获取图片的高和宽，img.shape返回一个元组,包含图像的维度信息。
img_gray=np.zeros([h,w],img.dtype)
#创建一张和当前图片大小一样的单通道图片，np.zeros是NumPy库中的一个函数,用于创建一个全零的数组。
# 手写过程：
for g in range(h):
#遍历每一行高以及每一行宽
    for k in range(w):
        m=img[g,k]
        #取出当前高和宽中的BGR坐标
        img_gray[g,k]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
        #将BGR转化成gray(灰度图)并复制给新的图像
        #采用浮点算法:Gray=R0.3+G0.59+B0.11
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#或者直接调用接口
cv2.imshow("grayscale image result",img_gray)
#显示灰度图像结果



img = plt.imread("lenna.png")
#使用matplotlib第三方库的imread来读取图片
plt.subplot(221)
#创建一个2x2的网格,并选择第一个子图作为当前绘图的区域
plt.imshow(img)
plt.title("Original image")
#设置当前子图的标题
plt.axis("off")
#关闭坐标轴显示

#灰度化
img_gray = rgb2gray(img)
#使用rgb2gray函数将原始图像img转换为灰度图像,并将结果存储在变量img_gray中
plt.subplot(222)
#选择第1行第2列作为当前要绘制的子图
plt.imshow(img_gray, cmap="gray")
#在当前子图中显示灰度图像img_gray,并使用灰度颜色映射(colormap)
plt.title("Grayscale image")
#设置当前子图的标题
plt.axis("off")
#关闭坐标轴显示
#
#二值化
rows, cols = img_gray.shape
#手写过程：
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5  ) :
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1
#或者调用接口：
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
#选择第2行第1列作为当前要绘制的子图
plt.imshow(img_gray, cmap="binary")
plt.title("Binary image")
#设置当前子图的标题
plt.axis("off")
#关闭坐标轴显示
plt.tight_layout()
# 自动调整每张图的间距
plt.show()

cv2.waitKey(0)






