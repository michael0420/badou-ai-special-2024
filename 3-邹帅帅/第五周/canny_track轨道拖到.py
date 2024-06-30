

'''
Canny边缘检测：优化的程序
'''

import cv2
import numpy as np

def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0) # 高斯滤波
    detected_edges = cv2.Canny(gray, 
                        lowThreshold, 
                        lowThreshold * ratio, 
                        apertureSize = kernel_size) # 边缘检测 kernel_size指定Sobel算子大小
                        
     #用原始颜色添加到检测的边缘上。 
     #按位“与”操作。对于每个像素,将两幅输入图像相应位置的像素值分别进行按位“与”运算,输出的结果图像的对应像素值即为这两幅输入图像对应像素值的按位与结果。
     #src1和src2表示要进行按位“与”操作的两幅输入图像；只有在两个图像对应位置的像素值都为非零时，该位置的像素值才为非零，否则为零
     #mask 是可选参数，如果指定了掩膜，则只对掩膜对应位置的像素进行按位“与”操作。函数的返回值表示按位“与”运算的结果。
                        
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    cv2.imshow("canny", dst)


# 设置全局变量
lowThreshold = 0
max_lowThreshold = 100
kernel_size = 3
ratio = 3


img = cv2.imread('lenna.png')
#gray = cv2.imread('lenna.png', 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#创建一个名为'canny result'的窗口用于显示图像。
cv2.namedWindow('canny result') 

"""   
cv2.createTrackbar(trackbarName, windowName, value, count, onChange)
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
"""

cv2.createTrackbar('Min threshold', 'canny result', lowThreshold,max_lowThreshold, CannyThreshold)

'''
添加一个名为'Min threshold'的轨迹条，它位于'canny result'窗口中，
允许用户调整低阈值。轨迹条的范围是从lowThreshold到max_lowThreshold，
默认值为lowThreshold，
并且每当轨迹条的值改变时，都会调用CannyThreshold函数。
'''

CannyThreshold(0) # 函数初始化
if cv2.waitKey(0) == 27:  # 在计算机键盘上，ASCII码27对应于Escape键（Esc）。所以，这段代码的意思是：“如果用户按下Esc键
    cv2.destroyALLWindows()


'''
函数初始化作用：
即时反馈：初始化显示可以让用户立即看到程序的效果，即便他们还没有开始手动调整阈值。这提供了即时的视觉反馈，让用户明白程序已经准备好，并且理解界面如何响应他们的操作。

预设配置检查：通过调用函数，可以确保函数逻辑正确无误，包括图像读取、灰度转换、Canny算法应用及结果显示等环节。如果在初始化阶段发现任何问题，如文件读取错误、参数设置不当等，可以及时调试修复。

'''
