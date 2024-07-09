import cv2
import numpy as np

img = cv2.imread("work.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 先声明一下
sift = cv2.SIFT_create()   # 使用了OpenCV库中的xfeatures2d模块中的SIFT算法。创建后，可以使用该对象进行关键点检测和描述符计算
keypoints, descriptor = sift.detectAndCompute(gray, None)     # None，表示没有预先指定掩模图像

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
'''
image=img这是要在其上绘制关键点的输入图像
outImage=img：这个参数表示绘制关键点后的输出图像，即将结果绘制在哪张图片上。指定为img，表示将结果绘制在原始图像上
keypoints=keypoints：这是要绘制的关键点的位置
绘制关键点时的风格标志。在这里，使用的是cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS，表示以丰富的风格绘制关键点
color=(51, 163, 236)：这是绘制的关键点的颜色，指定为BGR格式
'''
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

#img=cv2.drawKeypoints(gray,keypoints,img)

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
