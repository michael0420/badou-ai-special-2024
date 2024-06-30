import cv2
import numpy as np

# 读图
img1 = cv2.imread('iphone1.png')
img2 = cv2.imread('iphone2.png')

# 声明
sift = cv2.xfeatures2d.SIFT_create()
# 计算关键点及
keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
keypoints2, descriptor2 = sift.detectAndCompute(img2, None)
# 在两张原图上绘制出关键点
img1_sife = cv2.drawKeypoints(img1, keypoints=keypoints1, outImage=img1,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=None)
img2_sife = cv2.drawKeypoints(img2, keypoints=keypoints2, outImage=img2,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=None)
# 将两张带有关键点的图片放在一张图中
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
img_mix = np.zeros([max(h1, h2), w1 + w2, 3], np.uint8)
# print(img_mix.shape)
img_mix[:h1, :w1] = img1_sife
img_mix[:h2, w1:w1 + w2] = img2_sife
# 找到两张图的最佳匹配点
a = cv2.BFMatcher(cv2.NORM_L2)  # 声明：利用欧式距离匹配
matches = a.knnMatch(descriptor1, descriptor2, k=2)  # 图片1的特征与图片二的特征一一匹配，找到图1中每个关键点与图二相似度最高的前K个
print(matches)
# 找到匹配最佳的值（这里利用第一个匹配点和第二个匹配点的一般距离进行比较，如果小于其一半，说明其遥遥领先的匹配）
goodmatches = []

for m, n in matches:
    if m.distance < n.distance * 0.5:
        goodmatches.append(m)

# 找到最佳特征点分别在图片中的位置
point1 = [kpp.queryIdx for kpp in goodmatches[:30]]
point2 = [kpp.trainIdx for kpp in goodmatches[:30]]

# 对点连线
post1 = np.int32([keypoints1[pp].pt for pp in point1])
post2 = np.int32([keypoints2[pp].pt for pp in point2]) + (w1, 0)

for (x1, y1), (x2, y2) in zip(post1, post2):
    cv2.line(img_mix, (x1, y1), (x2, y2), color=(0, 255, 0))
# 出图
cv2.imshow('b', img2_sife)
cv2.imshow('a', img_mix)
cv2.waitKey(0)
