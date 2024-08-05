# -*- coding: utf-8 -*-
'''@Time: 2024/5/7 21:27

'''
import cv2
import numpy as np
img = cv2.imread("../lenna.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("canny",cv2.Canny(gray,30,200))
cv2.waitKey()
