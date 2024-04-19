import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1.灰度图进行直方图均衡化
img = cv2.imread('city.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.imread('city.jpg', 0)  # 读图时直接进行0灰度化,1默认彩色，-1透明度通道
Histogram = cv2.equalizeHist(img_gray)  # equalizeHist(src, dst=None) src：图像矩阵(单通道图像)
cv2.imshow('Histogram', np.hstack([img_gray, Histogram]))  # hstack对象为tuple,[]合并矩阵再转为元组，灰度图与均衡化后的图片对比，源图为三通道，不能并列显示
cv2.waitKey()

# 2.彩色图进行直方图均衡化。cv2.split和cv2.merge进行通道的拆解合并
img = cv2.imread('city.jpg')
b, g, r = cv2.split(img)  # 拆解成三个单通道图像矩阵，进行直方图均衡化
# print(b)
bH, gH, rH = cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)
cv2.namedWindow('b, g, r', cv2.WINDOW_NORMAL)  # 拼接图片完整显示
cv2.imshow('b, g, r', np.hstack([bH, gH, rH]))
cv2.waitKey()

h, w, c = img.shape
zeros = np.zeros([h, w], dtype=np.uint8)
img_bH = cv2.merge([zeros, zeros, bH])
img_gH = cv2.merge([zeros, gH, zeros])
img_rH = cv2.merge([rH, zeros, zeros])
cv2.imshow('Simpl_Color_img', np.hstack([img_rH, img_gH, img_bH]))  # 另外两通道为0的合并显示
cv2.waitKey()

img_color = cv2.merge((bH, gH, rH))  # 三通道合并
cv2.imshow('Color_img & Color_Histogram', np.hstack([img, img_color]))
cv2.waitKey()

# 3.plt.hist绘制像素直方图
img = cv2.imread('city.jpg')
# img1 = cv2.imread('lenna.png', 0)  # 以灰度图读入，和skimage,cvtcolor,公式法都有区别，不严格区分对错
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
img_histo = cv2.equalizeHist(img_gray)  # 均衡化

Histogram = cv2.calcHist([img_histo], [0], None, [256], [0, 255])
plt.subplot(1, 3, 1)
plt.plot(Histogram)
'''
# opencv获取直方图的接口，可直接绘图。参数：[图名]，[通道数]，通幅None，[范围]，[像素范围0,255]
'''
# print(Histogram)
# print(Histogram[50])
# print(type(Histogram))
plt.subplot(1, 3, 2)
plt.hist(img_gray.ravel(), 256)  # 均衡化前后灰度直方图对比,bins为直方图条数
plt.subplot(1, 3, 3)
plt.hist(img_histo.ravel(), 256)  # x.ravel（）将矩阵展平，且改变原数值。flatten不改变原数值
plt.show()

# 4.opencv接口获取直方图，绘制像素直方图
img = cv2.imread('lenna.png')
b, g, r = cv2.split(img)
bH, gH, rH = cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)  # 均衡化

b_histo = cv2.calcHist([b], [0], None, [256], [0, 255])
g_histo = cv2.calcHist([g], [0], None, [256], [0, 255])
r_histo = cv2.calcHist([r], [0], None, [256], [0, 255])
plt.plot(b_histo, color='blue')
plt.plot(g_histo, color='green')
plt.plot(r_histo, color='red')
plt.show()
