# -*- coding: gbk -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

def equalHist(img):
    # 实现直方图均衡化算法
    # 步骤：
    # 依次扫描原始灰度图像的每一个像素，计算出图像的灰度直方图H
    # 计算灰度直方图的累加直方图
    # 根据累加直方图和直方图均衡化原理得到输入与输出之间的映射关系。
    # 最后根据映射关系得到结果：dst(x, y) = H'(src(x,y))进行图像变换
    # 写出代码实现上述算法
    # 1. 计算直方图
    # 第一个参数[dst]是要计算直方图的图像列表。在这里，只计算了一个图像dst的直方图。
    # 第二个参数[0]是要计算的通道的索引列表。在这里，只计算了第一个通道（索引为0）的直方图。对于灰度图像，只有一个通道，所以这里是[0]。对于彩色图像，有三个通道（红、绿、蓝），如果要计算所有通道的直方图，这里应该是[0, 1, 2]。
    # 第三个参数None是一个掩模（mask）。如果提供了掩模，则只计算掩模像素点的直方图。在这里，没有提供掩模，所以计算的是整个图像的直方图。
    # 第四个参数[256]是直方图的大小，也就是直方图中有多少个bin。在这里，直方图的大小是256，所以每个bin代表一个像素值的范围。
    # 第五个参数[0,256]是像素值的范围。在这里，像素值的范围是从0到256，所以这个直方图包括了所有可能的像素值。
    # 所以，hist = cv2.calcHist([dst],[0],None,[256],[0,256])这行代码的意思是，计算图像dst的直方图，直方图的大小是256，像素值的范围是从0到256，然后将结果存储在hist中。

    # 依次扫描原始灰度图像的每一个像素
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # 计算灰度直方图的累加直方图
    cdf = hist.cumsum()
    # 根据累加直方图和直方图均衡化原理得到输入与输出之间的映射关系
    # 计算映射表
    table = np.zeros(256)
    for i in range(256):
        table[i] = np.uint8(255 * cdf[i] / cdf[255])
    # 根据映射表得到结果
    dst = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dst[i, j] = table[img[i, j]]

    return dst
# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
# dst = cv2.equalizeHist(gray)
dst1 = equalHist(gray)

plt.plot(dst1)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst1])) # 并排显示原图和均衡化后的图像
cv2.waitKey(0)



# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = equalHist(b)
gH = equalHist(g)
rH = equalHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))

# 分别计算每个通道的直方图
# hist_b = cv2.calcHist([result],[0],None,[256],[0,256])
# hist_g = cv2.calcHist([result],[1],None,[256],[0,256])
# hist_r = cv2.calcHist([result],[2],None,[256],[0,256])

# 画出每个通道的直方图,用三个子图分别画出来
plt.subplot(311)
plt.plot(bH, color='b')
plt.subplot(312)
plt.plot(gH, color='g')
plt.subplot(313)
plt.plot(rH, color='r')
plt.show()


cv2.imshow("dst_rgb", result)
cv2.waitKey(0)
