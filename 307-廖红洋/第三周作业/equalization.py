# -*- coding: utf-8 -*-
"""
@File    :   equalization.py
@Time    :   2024/04/08 15:19:58
@Author  :   廖红洋 
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

gray = cv2.imread("lenna.png", 0)  # 参数为0得到灰度图

dst = cv2.equalizeHist(gray)  # 直方图均衡化，这里得到的dst是一个均衡化之后的灰度图

hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
plt.figure()
plt.hist(gray.ravel(), 256)
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imwrite("lenna_equal.png", dst)  # 保存均衡化以后的图片
