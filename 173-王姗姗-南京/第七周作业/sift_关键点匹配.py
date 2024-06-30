import cv2

# 获取图片信息
img1 = cv2.imread('lenna.png')

# 图片灰度化
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# 创建sift特征检测器
sift = cv2.xfeatures2d.SIFT_create()
# 使用sift检测灰度图中的关键点和描述符，并返回关键点和描述符矩阵
kp1, des1 = sift.detectAndCompute(img1_gray, None)

# 对图像中的每个关键点绘制圆圈和方向
img = cv2.drawKeypoints(img1_gray, kp1, img1_gray,
                  color=(51, 163, 236),
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 显示处理后的图片
cv2.imshow('SIFT', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
