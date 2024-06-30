import numpy as np
import cv2
from matplotlib import pyplot as plt


def cannny(lowthreshold):
    detect_edges = (cv2.Canny(gray,
                              lowthreshold,
                              lowthreshold * ratio,
                              # 默认是kernel_size=5
                              apertureSize=kernel_size)
                    )

    # 对图像进行按位与操作
    dst = cv2.bitwise_and(img, img, mask=detect_edges)
    # 显示图像
    cv2.imshow('canny result', dst)


lowthreshold = 0
maxthreshold = 200
ratio = 3
kernel_size = 3

img = cv2.imread('lenna.png')
# 转换彩色图像为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 创建窗口
cv2.namedWindow('canny result')

# 创建轨迹条
cv2.createTrackbar('Min threshold:', 'canny result', lowthreshold, maxthreshold, cannny)

# 初始化,调用函数
cannny(0)

# 等待按键事件
cv2.waitKey(0)
cv2.destroyAllWindows()
