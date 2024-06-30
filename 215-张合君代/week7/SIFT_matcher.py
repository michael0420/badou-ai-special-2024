import cv2
import numpy as np

# 读取两幅图像
img1 = cv2.imread('form-LNADHAB22P1057441-1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('form-LNADHAB22P1057441-2.jpg', cv2.IMREAD_GRAYSCALE)

# 检查图像是否加载成功
if img1 is None or img2 is None:
    print("Error loading images")
    exit()

# 创建SIFT对象
sift = cv2.SIFT_create()

# 检测并计算特征点和描述子
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 使用FLANN进行最近邻搜索
index_params = dict(algorithm=1, trees=5)  # 使用KD树
search_params = dict(checks=50)  # 指定搜索次数
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 进行特征点匹配
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 比率测试，筛选出好的匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 计算相似度百分比
similarity = len(good_matches) / min(len(keypoints1), len(keypoints2)) * 100
# 绘制匹配结果
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
print(similarity)

# 显示匹配结果
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)  # 创建一个可以调整大小的窗口
cv2.resizeWindow('Image', 800, 600)
cv2.imshow('Image', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
