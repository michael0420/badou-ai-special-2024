# coding = utf-8

'''
        实现SIFT关键点描述
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图片并灰度化
lenna = cv2.imread('lenna.png')
gray = cv2.cvtColor(lenna, cv2.COLOR_BGR2GRAY)

# 声明sift函数，注意知识产权合规问题
# 执行，获取关键点与描述
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

# 绘图表示关键点位置、尺度、方向信息
res = cv2.drawKeypoints(image=lenna, keypoints=kp, outImage=None, color=(11, 158, 136),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('sift_kp', res)
cv2.waitKey(0)
