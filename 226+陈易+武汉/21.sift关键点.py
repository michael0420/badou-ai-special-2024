import cv2
import numpy as np

img = cv2.imread("lenna.png")                 # 读图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 灰度化

# sift为实例化的sift函数
sift = cv2.xfeatures2d.SIFT_create()
# 提取出具有尺度不变性、旋转不变性和部分视角不变性的特征点和特征描述子。
keypoints, descriptor = sift.detectAndCompute(gray, None)   # keypoints：关键点。 descriptor：特征描述子

"""绘制特征点函数cv2.drawKeypoint()
image:也就是原始图片
keypoints：从原图中获得的关键点，这也是画图时所用到的数据
outputimage：输出              //可以是原始图片 
color：颜色设置，通过修改（b,g,r）的值,更改画笔的颜色，b=蓝色，g=绿色，r=红色。
flags：绘图功能的标识设置
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
"""
img = cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,color=(51,163,236),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('sift_keypoints',img)
cv2.waitKey()