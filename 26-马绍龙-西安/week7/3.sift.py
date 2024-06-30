import cv2

# 从文件中读取图像
img1 = cv2.imread("mount1.png")

# 将图像转换为灰度图以供SIFT算法使用
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# 创建SIFT对象
sift = cv2.xfeatures2d.SIFT_create()

# 检测关键点和计算描述符
keypoints, descriptor = sift.detectAndCompute(gray, None)

# 在原图上绘制关键点，cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
img1 = cv2.drawKeypoints(image=img1, outImage=img1, keypoints=keypoints,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(51, 163, 236))

# 显示带有关键点标注的图像
cv2.imshow('img1', img1)
cv2.waitKey(0)  # 等待按键，按键后关闭窗口
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
