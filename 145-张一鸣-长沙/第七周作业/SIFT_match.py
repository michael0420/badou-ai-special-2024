# coding = utf-8

'''
        实现SIFT关键点描述
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

def pic(img_1, img_2, kp1, kp2, match_cp):
    print(img_1.shape)
    print(img_2.shape)
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]

    # 生成新图片合并图像，设定展示位置
    res = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    res[:h1, :w1] = img_1
    res[:h2, w1:w1 + w2] = img_2
    # 提取匹配的关键点索引
    p1 = [kp.queryIdx for kp in match_cp]
    p2 = [kp.trainIdx for kp in match_cp]
    # 获取匹配的关键点位置，并转换为整数坐标
    post1 = np.int32([kp1[p].pt for p in p1])
    post2 = np.int32([kp2[p].pt for p in p2])

    for (x1, y1), (x2, y2) in zip(post1, post2):
        x2 += w1    # 将第二张图像的坐标调整为在结果图像中的位置
        # 使用cv2画线连接关键点，使用BGR颜色和线宽
        cv2.line(res, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('sift_and_match', res)

# 读取图片
img_1 = cv2.imread('iphone1.png')
img_2 = cv2.imread('iphone2.png')

# 声明SIFT函数
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img_1, None)
kp2, des2 = sift.detectAndCompute(img_2, None)

# 全量遍历匹配相似度最高的前k个，NORM_L2欧式距离匹配
match = cv2.BFMatcher(cv2.NORM_L2)
matches = match.knnMatch(des1, des2, k=2)

# 关键点匹配组
match_cp = []
for m, n in matches:
    # 如果其中一匹配点的距离小于另一点距离的一半
    # 说明此点相似度较优
    if m.distance < n.distance * 0.5:
        match_cp.append(m)
print(match_cp)

# 调用方法
pic(img_1, img_2, kp1, kp2, match_cp[:10])

cv2.waitKey(0)
cv2.destroyAllWindows()

