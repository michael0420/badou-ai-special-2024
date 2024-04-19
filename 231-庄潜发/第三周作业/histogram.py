"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/4/7 22:27
"""
"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/4/6 23:25
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


def gray():
    # 灰度化
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 灰度图直方图灰度化
    dst_img = cv2.equalizeHist(gray_img)

    # 直方图展示, 参数---> 图像列表, 通道索引, mask掩码(没有用到,None), 直方图大小(直方图有多少柱形),  直方图像素值范围0-256不包含256
    # hist = cv2.calcHist([dst_img], [0], None, [256], [0, 256])
    #
    # plt.plot(hist)
    # plt.show()

    plt.figure()  # 创建一个新的matplotlib图形窗口
    # plt.hist(dst_img.ravel(), 256)  # hist函数绘制直方图图像,ravel()将数据扁平会为一维数组,256为直方图柱形的数量
    plt.hist(dst_img.ravel(), 256)  # 原图灰度直方图
    plt.show()  # 显示图形

    # np.hstack()函数,水平堆叠, 将原始灰度图 和 均衡化的直方图进行水平拼接一个新图像, 并排显示进行比较
    cv2.imshow("Histogram Equalization", np.hstack([gray_img, dst_img]))
    cv2.waitKey()


def rgb():
    # 彩色图像均衡化需要分解通道, 对每个通道均衡化
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    # cv2.imshow('dit_rgb', result)
    cv2.imshow('dst_rgb', np.hstack([img, result]))


if __name__ == '__main__':
    # 获取原图
    img = cv2.imread("lenna.png")
    gray()
    rgb()
    cv2.waitKey()
