"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/5/26 20:34
"""
import cv2

# 读取图片
img = cv2.imread("../lenna.png")
if img is None:
    print("Error: Image not found.")
    exit()

# 转换到灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建SIFT对象
sift = cv2.xfeatures2d.SIFT_create()

# 检测关键点和描述符, 每个关键点都有唯一的描述符
keypoints, descriptor = sift.detectAndCompute(gray, None)
print(keypoints, descriptor)

# 绘制关键点,
# image=指定要绘制关键点的图片
# keypoints=提供关键点列表
# outImage=None 创建一个新图像用于绘制结果
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 除了绘制其位置外，还会绘制关键点的大小和方向
# color=(51, 163, 236): 指定了绘制关键点时使用的颜色
img_with_keypoints = cv2.drawKeypoints(image=gray, keypoints=keypoints, outImage=None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                       color=(51, 163, 236))

# 显示带有SIFT关键点的图像
cv2.imshow('sift_keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
