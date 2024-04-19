import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
直方图均衡化使用函数接口：equalizeHist(src,dst=None)
src: 图像矩阵（单通道图像）
dst:默认即可
'''

# 获取灰度图像：
img = cv2.imread("lenna.png",1)
'''
第二个参数是一个标志，用于指定读取图像的方式。参数 1 表示以彩色图像的方式加载图像。
这意味着图像将以彩色模式加载，包含红色、绿色和蓝色通道的信息。如果将标志设置为 0，则表示以灰度图像的方式加载图像，只包含灰度信息。

!!! 如果不指定第二个参数，cv2.imread() 默认会以彩色图像的方式加载图像，与指定 1 效果相同。因此，如果不写 1，则默认以彩色图像的方式加载名为 "lenna.png" 的图像。
'''
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 灰度直方图均衡化：
dst = cv2.equalizeHist(gray)

# 直方图：
hist = cv2.calcHist([dst],[0],None,[256],[0,256])


plt.figure()
plt.hist(dst.ravel(),256)
plt.show()
'''
plt.hist()：plt绘制直方图的函数。dst.ravel()：将目标图像 dst 转换为一维数组，以便将所有像素值作为一个序列进行处理。.ravel() 方法用于将多维数组展平为一维数组。
256：指定直方图的 bin 的数量，即将像素值范围划分成多少个区间。在这里，将像素值范围划分成 256 个区间，每个区间代表一个像素值。
'''

cv2.imshow('Histogram Equlization',np.hstack([gray,dst]))
cv2.waitKey(0)
'''
np.hstack([gray, dst]) 将灰度图像 gray 和经过直方图均衡化处理后的图像 dst 水平堆叠起来。np.hstack() 函数用于沿水平方向将图像堆叠在一起。

cv2.imshow('Histogram Equlization', np.hstack([gray, dst])) 的作用是在一个窗口中显示原始灰度图像和经过直方图均衡化处理后的图像，使用户可以比较它们的差异。

参数 0 表示等待时间为无限长，即程序会一直等待，直到用户按下键盘上的任意键。
'''





#-----------------------彩色图像的直方图均衡化------------------------------------------------------------------------------







'''
彩色图像的直方图均衡化：
'''

img = cv2.imread("lenna.png")
cv2.imshow("src",img)
cv2.waitKey(0)

# 彩色图像均衡化：需要分解通道，对每一个通道均衡化：
(b,g,r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# 然后再合并每一个通道均衡化之后的值：
result = cv2.merge((bH,gH,rH))
cv2.imshow("dst_rgb",result)
cv2.waitKey(0)