# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:47:29 2024

@author: DELL
"""

import cv2
import numpy as np
import pysnooper


@pysnooper.snoop()
def function(img, argument_1):
    hight, widht, channels = img.shape
    scale = int(hight / widht)
    argument_2 = int(argument_1 / scale)
    empty_image = np.zeros((argument_1, argument_2, channels), np.uint8)

    sh = argument_1 / hight
    sw = argument_2 / widht
    for i in range(argument_1):
        for j in range(argument_2):
            a = i/sh + 0.5
            b = j/sw + 0.5
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            # if i > argument_1-2:
            if i >= 1023:
                if j % 3 == 0:
                    end_str = "\n  "   # 换行
                else:
                    end_str = "\t  "   # 制表符
                print("i:{:.2f}".format(i), end="\t")
                print("j:{:.2f}".format(j), end="\t")
                print("a:{:.2f}".format(a), end="\t")
                print("b:{:.2f}".format(b), end="\t")
                print("x:{:.2f}".format(x), end="\t")
                print("y:{:.2f}".format(y), end=end_str)
            empty_image[i, j] = img[x, y]
    return empty_image


lenna = cv2.imread("lenna.png")
hight, widht, channels = lenna.shape

expand_img = function(lenna, 1024)   # 可输入 需要放大的图片高度，自适应宽度，比例不变

# cv2.imshow("original lenna", lenna)
# cv2.imshow("nearest interp lenna", expand_img)
# cv2.waitKey(0)
