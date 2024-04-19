import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
equalizeHist代表直方图均衡化
src 是输入图像矩阵，必须是单通道图像（例如，灰度图像）
dst 是输出图像矩阵
cv2.equalizeHist(src, dst=None)是函数的原型
'''


#获取灰度图像
img = cv2.imread("C:/Users/86188/Pictures/lenna.png",1)  #读图
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

#直方图计算
'''
cv2.calcHist() 函数用于计算图像的直方图
cv2.calcHist() 函数的语法：cv2.calcHist(images, channels, mask, histSize, ranges[, accumulate])
images 是输入图像列表。在这个例子中，它是一个包含直方图均衡化后的图像 dst 的列表
channels 是输入图像的通道列表。在这个例子中，它是一个包含通道索引 0 的列表，表示我们只计算第一个通道（灰度通道）的直方图
mask 是用于选择输入图像中特定区域的掩码图像。在这个例子中，它被设置为 None，表示我们不使用掩码图像
histSize 是每个通道的灰度级别数。在这个例子中，它是一个包含值 256 的列表，表示我们计算 256 个灰度级别的直方图
ranges 是每个通道的灰度级别范围。在这个例子中，它是一个包含值 [0, 256] 的列表，表示我们计算的灰度级别范围是从 0 到 256
accumulate 是一个可选参数，用于指定是否累加多个图像的直方图。在这个例子中，它被省略
'''

hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(),256)
plt.show()
#np.hstack() 函数用于水平拼接两个数组。它用于将原始灰度图像和直方图均衡化后的图像水平拼接在一起
cv2.imshow("Histogram Equalization",np.hstack([gray,dst]))
cv2.waitKey(0)


'''
#彩色图像直方图均衡化
img = cv2.imread("C:/Users/86188/Pictures/lenna.png",1)
cv2.imshow("src",img)

#彩色图像均衡化，需要分解通道，每个通道均衡化
(b,g,r) = cv2.split(img)
#cv2.split() 函数将彩色图像分解为三个通道：蓝色、绿色和红色
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

#合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_bgr", result)

cv2.waitKey(0)
'''








