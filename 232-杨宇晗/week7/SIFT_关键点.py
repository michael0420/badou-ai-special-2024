import cv2
import numpy as np

# 读取图像
img = cv2.imread("shangri-la.jpg")
# 转换到灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建SIFT对象，注意这里直接使用cv2.SIFT_create()
sift = cv2.SIFT_create()
# 检测关键点和计算描述符
keypoints, descriptor = sift.detectAndCompute(gray, None)

# 在图像上绘制关键点
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

# 显示图像
cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
