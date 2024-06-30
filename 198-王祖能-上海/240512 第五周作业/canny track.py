'''
Canny边缘检测，带调节杆，调整上下阈值
'''

import cv2
import numpy as np


def Canny_Threshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(img, [3, 3], sigmaX=0, sigmaY=0)  # 先进行高斯滤波
    '''
    cv2.GaussianBlur（ SRC，ksize，sigmaX [，DST [，sigmaY [，borderType ] ] ] ） →DST
    src –输入图像；图像可以具有任何数量的信道，其独立地处理的，但深度应CV_8U，CV_16U，CV_16S，CV_32F或CV_64F。
    ksize –高斯核大小。 ksize.width 并且 ksize.height 可以有所不同，但它们都必须是正数和奇数。
    sigmaX – X方向上的高斯核标准偏差。
    dst –输出与图像大小和类型相同的图像src。
    sigmaY – Y方向上的高斯核标准差；如果 sigmaY 为零，则将其设置为等于 sigmaX；如果两个西格玛均为零，则分别根据ksize.width 和 进行计算 ksize.height；
    borderType –像素外推方法。
    '''
    detected_edges = cv2.Canny(img, lowThreshold, ratio * lowThreshold, apertureSize=kernel_size, L2gradient=True)
    img_new = cv2.bitwise_and(img_color, img_color, mask=detected_edges)  # 只对mask掩膜相应位置进行按位与
    '''
    #用原始颜色添加到检测的边缘上。 
    #按位“与”操作。对于每个像素,将两幅输入图像相应位置的像素值分别进行按位“与”运算,输出的结果图像的对应像素值即为这两幅输入图像对应像素值的按位与结果。
    #src1和src2表示要进行按位“与”操作的两幅输入图像；
    #mask 是可选参数，如果指定了掩膜，则只对掩膜对应位置的像素进行按位“与”操作。函数的返回值表示按位“与”运算的结果。
    '''
    cv2.imshow('Canny', img_new)


if __name__ == '__main__':
    img_color = cv2.imread('lenna.png', 1)
    img = cv2.imread('lenna.png', 0)

    lowThreshold = 0
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3
    '''
    cv2.createTrackbar()有5个参数
    trackbarName是这个trackbar名称；windowName是trackbar所在面板名称；value是trackbar默认值,也是调节的对象
    count是是这个trackbar上调节的范围(0~count)；onChange是调节trackbar时调用的回调函数名
    '''
    cv2.namedWindow('Canny', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Canny Track', 'Canny', lowThreshold, max_lowThreshold, Canny_Threshold)
    Canny_Threshold(0)  # initialization
    if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2  实际按任意键都可以退出？
        cv2.destroyAllWindows()
