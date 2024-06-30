import cv2
import numpy as np

img = cv2.imread("D:\cv_workspace\picture\lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

import cv2

# 检查 xfeatures2d 属性是否存在
# if 'xfeatures2d' in dir(cv2):
#     print("xfeatures2d 模块已安装并可用。")
# else:
#     print("xfeatures2d 模块未安装或不可用。")
#sift = cv2.xfeatures2d.SIFT_create()

'''
创建一个 SIFT 特征检测器，返回一个 SIFT 特征检测器对象
'''
sift = cv2.SIFT.create()
'''
函数的参数有两个：
gray：输入的灰度图像，通常是 8 位或 16 位的整数图像。如果输入是彩色图像，可以先用 cv2.cvtColor() 函数转换为灰度图像。
None：特征描述子的类型，默认为 cv2.SIFT_DEFAULT，表示使用默认的 SIFT 特征描述子。
                                           你也可以选择其他类型的特征描述子，例如 cv2.SIFT_128 表示使用 128 维的 SIFT 特征描述子。
函数会返回两个值：
keypoints：一个 numpy.ndarray 对象，包含了检测到的 SIFT 特征点的坐标。每个特征点的坐标用 (x, y) 表示，其中 x 和 y 分别是特征点在图像中的横坐标和纵坐标。
           坐标点 (20.486684799194336, 74.56652069091797)
descriptor：一个 numpy.ndarray 对象，包含了检测到的 SIFT 特征点的特征描述子。每个特征点的特征描述子是一个 128 维的向量，可以用于特征匹配等后续操作。
           shape(1100,128)
'''
keypoints, descriptor = sift.detectAndCompute(gray, None)  # 返回 检测到的 SIFT 特征点的坐标和特征描述子

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
'''
OpenCV 库的`drawKeypoints()`函数在图像`img`上绘制检测到的特征点`keypoints`。函数的参数有以下几个：
1. `image`：输入图像，这里是`img`。
2. `outImage`：输出图像，这里也是`img`，表示在原始图像上绘制特征点。
3. `keypoints`：特征点列表，这里是`keypoints`。
4. `flags`：绘制特征点的标志，这里是`cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS`，表示绘制带有方向和大小的特征点。
5. `color`：特征点的颜色，这里是`(51, 163, 236)`，表示用此绘制特征点。
所以，这个代码会在图像`img`上绘制检测到的特征点`keypoints`，每个特征点用蓝色表示，并带有方向和大小。
'''
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))
                        
#img=cv2.drawKeypoints(gray,keypoints,img)

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
