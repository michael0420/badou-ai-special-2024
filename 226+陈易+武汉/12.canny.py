# encoding = UTF-8

import cv2
import numpy as np

"""
cv2.Canny(image, threshold1, threshold2,[,edges[,aperureSize[,L2gradient ]]])
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是阈值1；
第三个参数是阈值2。
"""

img = cv2.imread("lenna.png",1)                 # 1是彩色图  bgr   # 0是灰度图
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("canny",cv2.Canny(gray,50,150))
cv2.waitKey()
cv2.destroyAllWindows()