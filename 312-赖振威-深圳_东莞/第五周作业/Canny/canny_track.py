#!/usr/bin/env python
# encoding=gbk
# 上面两行是指定 Python 解释器以及设置文件编码

'''
Canny边缘检测：优化的程序
'''

import cv2
import numpy as np 

# 定义 CannyThreshold 函数，用于执行 Canny 边缘检测
def CannyThreshold(lowThreshold):
    # 对灰度图像进行高斯模糊处理
    detected_edges = cv2.GaussianBlur(gray,(3,3),0) # 高斯滤波
    print(detected_edges)
    # 显示结果
    cv2.imshow('detect_edges', detected_edges)
    '''
    [[162 162 162 ... 167 152 142]
     [162 162 162 ... 167 152 142]
     [162 162 162 ... 167 152 142]
     ...
     [ 44  46  49 ... 102 101 100]
     [ 44  46  50 ... 103 104 105]
     [ 44  47  51 ... 103 106 107]]
    '''

    # 调用 cv2.Canny() 函数执行 Canny 边缘检测
    detected_edges = cv2.Canny(detected_edges,
            lowThreshold,
            lowThreshold*ratio,
            apertureSize = kernel_size)  #边缘检测
    print(detected_edges)  # 边缘检测矩阵
    # 显示结果
    cv2.imshow('detected_edges', detected_edges)
    '''
    [[  0   0   0 ...   0 255   0]
     [  0   0   0 ...   0 255   0]
     [  0   0   0 ...   0 255   0]
     ...
     [  0 255   0 ...   0   0 255]
     [  0   0 255 ...   0   0   0]
     [  0 255   0 ...   0 255   0]]
    这段代码是在调用 OpenCV 的 Canny 边缘检测函数 cv2.Canny()。让我逐行解释：
    detected_edges = cv2.Canny(detected_edges,：这一行代码调用了 cv2.Canny() 函数，它的第一个参数是 detected_edges，这是一个经过高斯模糊处理后的图像，通常是灰度图像，用于进行边缘检测。
    lowThreshold,：这是 Canny 边缘检测的低阈值参数，它是一个整数，用于控制边缘检测的灵敏度。像素梯度低于低阈值的边缘点会被抛弃。
    lowThreshold * ratio,：这里的 ratio 是一个系数，通常设置为 3，用于计算 Canny 边缘检测的高阈值。高阈值是低阈值的倍数，用于确定强边缘的阈值。这个系数的选择是根据经验和具体应用场景来调整的。
    apertureSize=kernel_size)：apertureSize 参数指定 Sobel 算子的孔径大小，它用于计算图像的梯度。kernel_size 是一个奇数，通常设置为 3，表示使用 3x3 的 Sobel 算子进行梯度计算。
    综合起来，这一行代码的作用是调用 Canny 边缘检测算法，并传递了低阈值、高阈值、和 Sobel 算子的孔径大小等参数，用于在经过高斯模糊处理后的图像上执行边缘检测。
    '''
 
    # just add some colours to edges from original image.
    # 将检测到的边缘叠加在原始图像上
    dst = cv2.bitwise_and(img,img, mask=detected_edges)  # 用原始颜色添加到检测的边缘上
    print(dst)
    # 显示结果
    cv2.imshow('canny demo',dst)
    '''
    这段代码使用了 OpenCV 的 cv2.bitwise_and() 函数，让我解释一下：
    dst = cv2.bitwise_and(img,img,mask = detected_edges)：这行代码的作用是将检测到的边缘叠加在原始图像上，生成一个新的图像 dst。
    img：这是原始的彩色图像，即输入图像。
    mask = detected_edges：这里的 detected_edges 是经过 Canny 边缘检测后得到的边缘图像，它是一个二值图像，其中边缘像素值为非零，非边缘像素值为零。
    在 cv2.bitwise_and() 函数中，mask 参数指定了一个掩码，它决定了在哪些位置进行位与运算。
    具体来说，这个掩码会将原始图像中的像素值与 detected_edges 中的像素值进行逐像素的位与运算。只有当掩码像素值为非零时，才会保留原始图像中对应位置的像素值，否则置零。这样就实现了将边缘叠加在原始图像上的效果。
    综合起来，这行代码的作用是将经过 Canny 边缘检测后得到的边缘图像叠加在原始彩色图像上，生成一个新的图像 dst，其中只保留了边缘部分。
    '''

# 初始化参数
lowThreshold = 0
max_lowThreshold = 100  
ratio = 3  
kernel_size = 3

# 读取图像并转换为灰度图像
img = cv2.imread('lenna.png')  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图

# 创建窗口用于显示结果
cv2.namedWindow('canny demo')  
  
#设置调节杠,
'''
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''
# 创建滑动条用于调节低阈值参数
cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)

# 初始化边缘检测
CannyThreshold(0)  # initialization

# 等待按下 ESC 键退出程序
if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2
    cv2.destroyAllWindows()  
