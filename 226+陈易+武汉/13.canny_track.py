# encoding = UTF-8

'''
Canny边缘检测：优化的程序
'''

import cv2
import numpy as np

def CannyThreshold(lowThreshold):
    # detected_edges = cv2.GaussianBlur(gray,(3,3),0)   # 高斯滤波
    detected_edges = cv2.Canny(gray,
                               lowThreshold,
                               lowThreshold * ratio,
                               apertureSize = kernel_size) # 边缘检测,apertureSize默认是5
    # 用原始颜色添加到检测的边缘上
    # 按位“与”操作。对每个像素，将两幅输入图像相应位置的像素值分别进行按位“与”运算，输出的结果图像的对应项数值即为这两幅输入图像对应像素值按位 与 结果
    # src1和src2表示要进行按位“与”操作的两幅输入图像；
    # mask 是可选参数，如果指定了掩膜，则只对掩膜对应位置的像素进行按位“与”操作，函数的返回值表示按位“与”运算的结果
    dst = cv2.bitwise_and(img,img,mask= detected_edges)
    cv2.imshow('canny result',dst)
    cv2.waitKey()

lowThreshold = 0            # 低阈值
max_lowThreshold = 100      # max低阈值
ratio = 3
kernel_size = 3             # kernel大小为：3*3

img = cv2.imread("lenna.png")                   # 读图
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     # 转灰度

cv2.namedWindow('canny result')         # cv2.namedWindow创建一个窗口

# 设置调节杆
"""
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值，也是调节的对象
第四个参数，是这个trackbar上调节的范围（0~count）
第五个参数，是调节trackbar时调佣的回调函数名
"""
cv2.createTrackbar('min threshold','canny result',lowThreshold,max_lowThreshold,CannyThreshold)

CannyThreshold(50)          # 初始化
if cv2.waitKey(0) == 5:     # wait for ESC key to exit cv2
    cv2.destroyAllWindows()