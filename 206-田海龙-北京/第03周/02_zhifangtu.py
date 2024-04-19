# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

from Util import cv_imread,cv_set_titile
from subplot_info import SubplotInfo

import matplotlib as mpl

# 设置matplotlib默认字体为支持中文的字体
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体（或其他支持中文的字体）
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方框的问题

img_path = "206-田海龙-北京/第03周/img/lenna.png"

# subplot子图行列数
img_row=2
img_col=2

def get_img(img_path):
    """
    获取图像
    """
    img = cv_imread(img_path)
    return img  

def get_img_gray(img):
    """
    获取灰度图像
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def show_histogram_img():
    """
    显示原图的直方图
    """
    img=get_img(img_path)

    plt.gcf().canvas.manager.set_window_title("原图直方图")
    # 显示图像直方图，未区分RGB
    plt.hist(img.ravel(), 256)
    plt.title("原图直方图")    

    # 显示图像直方图，区分RGB
    img_RBG=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    b,g,r=cv2.split(img)

    subplot_info=SubplotInfo(img_row,img_col)
    plt.figure()
    plt.gcf().canvas.manager.set_window_title("原图RGB直方图")
    # 原图三通道直方图
    subplot_info.set_subplot()
    plt.hist(b.ravel(), 256)
    plt.title("b")
    subplot_info.set_subplot()
    plt.hist(g.ravel(), 256)
    plt.title("g")
    subplot_info.set_subplot()
    plt.hist(r.ravel(), 256)
    plt.title("r")

    plt.figure()
    plt.subplot(121)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)

    # 图像直方图均衡化
    img_b = cv2.equalizeHist(b)
    img_g = cv2.equalizeHist(g)
    img_r = cv2.equalizeHist(r)
    dst = cv2.merge([img_b, img_g, img_r])
    # 这里显示的效果不太对，均衡化前后颜色差别很大
    cv2.imshow("dst", dst)
    
    dst=cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
    plt.subplot(122)
    plt.imshow(dst)

    # print(dst[0])
    # print(img_b[0][:5])
    # print(img_g[0][:5])
    # print(img_r[0][:5])

    plt.show()

def show_histogram_gray():
    """
    显示灰度直方图
    """
    img = cv_imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image_gray", gray)

    # 灰度图直方图
    # 没有如下注释代码，也可以显示直方图，没啥用
    # hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    plt.figure()
    plt.gcf().canvas.manager.set_window_title("灰度直方图")
    subplot_info = SubplotInfo(1, 2)
    subplot_info.set_subplot()
    plt.hist(gray.ravel(), 256)

     # 灰度图像直方图均衡化
    dst = cv2.equalizeHist(gray)
    cv2.imshow("aaa", dst)
    cv_set_titile("aaa", newTitle='灰度图像直方图-均衡化')

    subplot_info.set_subplot()
    plt.hist(dst.ravel(), 256)
    plt.show()

def main():
    # RGB直方图均衡化
    show_histogram_img()

    # 灰度直方图均衡化
    show_histogram_gray()

    cv2.waitKey()

if __name__ == "__main__":
    main()

    