import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1、读取图片
iphone1 = cv2.imread("iphone1.png")
iphone2 = cv2.imread("iphone2.png")
lenna = cv2.imread("lenna.png")
# cv2.imshow("iphone1",iphone1)
# cv2.imshow("iphone2",iphone2)
# cv2.waitKey(0)

# 方法一：
# 初始化SIFT对象
sift = cv2.xfeatures2d.SIFT_create()
# 检测和计算关键点及描述符：
# 这里，kp是一个关键点对象列表，des是一个NumPy数组，包含与关键点对应的描述符。
# 这是尺寸不变特征提取的最重要的步骤，这个函数里面其实是做了很多东西的
kp1, des1 = sift.detectAndCompute(iphone1, None)
kp2, des2 = sift.detectAndCompute(iphone2, None)
kp3, des3 = sift.detectAndCompute(lenna, None)

# 绘制关键点，可视化显示，画图
iphone1_with_keypoints = cv2.drawKeypoints(iphone1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
iphone2_with_keypoints = cv2.drawKeypoints(iphone2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
lenna_with_keypoints = cv2.drawKeypoints(lenna, kp3, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, )
iphone1_with_keypoints = cv2.cvtColor(iphone1_with_keypoints, cv2.COLOR_BGR2RGB)
iphone2_with_keypoints = cv2.cvtColor(iphone2_with_keypoints, cv2.COLOR_BGR2RGB)
lenna_with_keypoints = cv2.cvtColor(lenna_with_keypoints, cv2.COLOR_BGR2RGB)

# cv2.imshow("SIFT KeyPoints",lenna_with_keypoints)
# cv2.waitKey(0)
# cv2.imshow("SIFT KeyPoints",iphone2_with_keypoints)
# cv2.waitKey(0)
# cv2.imshow("SIFT KeyPoints",lenna_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.figure()
plt.subplot(131)
plt.imshow(iphone1_with_keypoints)
plt.subplot(132)
plt.imshow(iphone2_with_keypoints)
plt.subplot(133)
plt.imshow(lenna_with_keypoints)
plt.show()


# 方法二
# 读取图片并进行灰度化
img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建sift对象
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
gray = cv2.drawKeypoints(gray, keypoints, gray, color=(51, 163, 236), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("gray", gray)
cv2.waitKey(0)
