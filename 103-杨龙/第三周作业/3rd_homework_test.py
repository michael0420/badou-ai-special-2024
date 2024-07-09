# -*- encoding:utf-8 -*-

__author__ = 'Young'

import myimage
import cv2
import MyImageUtil

'''
1. bilinear test
'''
lenna = myimage.MyImage('../Image/lenna.png')
bilinear_img = lenna.zoom((700, 700))
cv2.imshow('bilinear image', bilinear_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
2 histogram test
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
3 sobel test
'''
lenna = myimage.MyImage('../Image/lenna.png', 0)
x_img = lenna.detect_image_edge(direction='X')
y_img = lenna.detect_image_edge(direction='Y')
edge_img = lenna.detect_image_edge(direction='All')
# cv2.imshow('x_abs', x_img)
# cv2.imshow('y_abs', y_img)
# cv2.imshow('edge', edge_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
MyImageUtil.compare_equalized_image([x_img, y_img, edge_img])
