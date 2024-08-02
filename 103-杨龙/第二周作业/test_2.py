# -*- encoding:utf-8 -*-

__author__ = 'Young'

from 第二周作业 import myimage
import matplotlib.pyplot as plt
import cv2

"""
第二周作业：
1.实现灰度化和二值化 
2.实现最临近插值
"""

'''
1. 实现灰度化和二值化
Three kinds of grayscale ways: (1) invoke cv2 method, (2) invoke pyplot method, (3) self codes
(1) cv2.imread()，pyplot.imread()返回的都是numpy矩阵，但矩阵数据类型不同，cv2是整型，plt是浮点型
(2) cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 用的是0.299 * R + 0.587 * G + 0.114 * B 公式  (星火大模型的查询结果)
(3) pyplot的rgb2gray()用的是0.2989 * R + 0.5870 * G + 0.1140 * B (星火大模型的查询结果 ) - TBD 这里需要看下源码
'''
# 1.1 灰度化
lenna_img = myimage.MyImage('../Image/lenna.png')
cv2_gray_img = lenna_img.gray_image('cv2')  # use cv2 package
plt_gray_img = lenna_img.gray_image('pyplot')  # use matplotlib.pyplot
my_gray_img = lenna_img.gray_image()  # convert to gray by self code
# print(np.array_equal(cv2_gray_img, plt_gray_img))  # False cv2
# print(np.array_equal(plt_gray_img, my_gray_img))  # False
# print(cv2_gray_img[0,0])  # cv2 该如何可以做二值化？
# print(plt_gray_img[0,0])  #
# print(my_gray_img[0,0])   #
cv2.imshow('cv2', cv2_gray_img)
cv2.imshow('default', my_gray_img)
plt.imshow(plt_gray_img, cmap='gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

# 1.2 二值化
lenna = myimage.MyImage('../Image/lenna.png')
plt.imshow(lenna.binarize(), cmap='gray')
plt.show()


"""
 2. 实现最临近插值，对图像进行上下采样
"""
lenna = myimage.MyImage('../Image/lenna.png')
cv2.imshow('zoom in image', lenna.zoom((800, 800)))
cv2.imshow('zoom out image', lenna.zoom((300, 300)))
# show时图像变色
cv2.imshow('rgb image', cv2.cvtColor(cv2.imread('../Image/lenna.png'), cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()
