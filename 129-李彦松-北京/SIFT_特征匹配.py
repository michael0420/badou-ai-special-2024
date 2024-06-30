import cv2
import numpy as np
 
def drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch):
    h1, w1 = img1_gray.shape[:2] #获取图像1的高和宽
    h2, w2 = img2_gray.shape[:2] #获取图像2的高和宽
 
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8) #3通道图像
    vis[:h1, :w1] = img1_gray #将图像1的灰度图像复制到vis的左侧
    vis[:h2, w1:w1 + w2] = img2_gray #将图像2的灰度图像复制到vis的右侧，w1:w1 + w2表示横坐标的范围
 
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
 
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
 
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0,0,255))
 
    cv2.namedWindow("match",cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)
 
img1_gray = cv2.imread("iphone1.png")
img2_gray = cv2.imread("iphone2.png")
 
#sift = cv2.SIFT()
sift = cv2.SIFT_create()
#sift = cv2.SURF()
 
kp1, des1 = sift.detectAndCompute(img1_gray, None) #kp是关键点，des是描述符
kp2, des2 = sift.detectAndCompute(img2_gray, None)
 
# BFmatcher with default parms
bf = cv2.BFMatcher(cv2.NORM_L2)
#opencv中knnMatch是一种蛮力匹配
#将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。
#knnMatch返回的是一个列表，列表中的每个元素是一个DMatch对象，DMatch对象有以下属性：
#DMatch.distance - 描述符之间的距离。越小越好。
#DMatch.trainIdx - 目标图像中描述符的索引。
#DMatch.queryIdx - 查询图像中描述符的索引。
#DMatch.imgIdx - 目标图像的索引。
matches = bf.knnMatch(des1, des2, k = 2)

goodMatch = []
for m,n in matches:
    if m.distance < 0.50*n.distance: #设置阈值为0.5
        goodMatch.append(m) #将距离比率小于0.5的匹配点保存下来
drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch[:20])
 
cv2.waitKey(0)
cv2.destroyAllWindows()
