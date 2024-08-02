# -*- encoding:utf-8 -*-

__author__ = 'Young'

import myimage
import cv2
import MyImageUtil

"""
第三周作业
1.实现双线性插值 
2.实现直方图均衡化 
3.实现sobel边缘检测
"""


'''
1.实现双线性插值 
'''
lenna = myimage.MyImage('../Image/lenna.png')
bilinear_img = lenna.zoom((700, 700))
cv2.imshow('bilinear image', bilinear_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
2.实现直方图均衡化 
'''
lenna = myimage.MyImage('../Image/lenna.png')
lenna_hist = lenna.equalize_hist(equalize_channels=3)
MyImageUtil.compare_equalized_image([lenna.image, lenna_hist])
MyImageUtil.show_equalization_histogram(lenna_hist)
lenna.convert_to_gray_image()
lenna_gray_hist = lenna.equalize_hist()
MyImageUtil.compare_equalized_image([lenna.image, lenna_gray_hist])
MyImageUtil.show_equalization_histogram(lenna_gray_hist)

'''
3.实现sobel边缘检测
'''
lenna = myimage.MyImage('../Image/lenna.png', 0)
x_img = lenna.detect_image_edge(direction='X')
y_img = lenna.detect_image_edge(direction='Y')
edge_img = lenna.detect_image_edge(direction='All')
MyImageUtil.compare_equalized_image([x_img, y_img, edge_img])
