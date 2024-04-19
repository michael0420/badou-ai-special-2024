"""
作业：直方图均衡化
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram_equalization():
    """
    直方图均衡化
    :return:
    """
    # 获取灰度图像  cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道，可用1作为实参替代
    img = cv2.imread("lenna.png", 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image_gray", img_gray)

    # 灰度图像直方图均衡化
    dst = cv2.equalizeHist(img_gray)

    # 统计直方图
    # cv2.calcHist(images,channels,mask,histSize,ranges)
    # images:原图像格式为uint8或float32，当传入函数时应用[]括起来
    # channels：若是灰度图应传[0],若为彩色图应为[0][1][2] 对应着BGR
    # mask：统计整幅图像的直方图应为None
    # histSize：传递[256]标示全像素范围
    # ranges：像素值范围，常为[0, 256]
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

    # 二维数组 print(hist)
    plt.plot(hist)
    plt.show()

    # 创建图，subplot创建子图
    plt.figure()

    # 一维数组 print(hist)

    # 绘制直方图
    # data = dst.ravel() -> 将高级数组转为一维数组
    # bins = 256 -> 用于指定数据分成的区间数量
    plt.hist(dst.ravel(), 256)

    # 显示图
    plt.show()
    # np.hstack：用于将两个或更多的数组沿着水平轴（即列）连接起来
    cv2.imshow("Histogram Equalization", np.hstack([img_gray, dst]))
    cv2.waitKey(0)

def histogram_equalization_BGR():
    """
    彩色图像直方图均衡化
    :return:
    """
    img = cv2.imread("lenna.png", 1)
    cv2.imshow("src", img)

    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    cv2.imshow("dst_rgb", result)
    cv2.waitKey(0)

if __name__ == '__main__':
    histogram_equalization()
    histogram_equalization_BGR()
