import cv2
import numpy as np

def drawMatchesKnn_cv2(img1,kp1,img2,kp2,matches):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    vis = np.zeros((max(h1,h2),w1+w2,3),np.uint8) #三通道图
    #分别把两张图放进vis
    vis[:h1,:w1] = img1
    vis[:h2,w1:w1+w2] = img2

    #获取匹配点的位置
    p1 = [kp.queryIdx for kp in matches] #在原图上的关键点(kp1)的匹配点索引
    p2 = [kp.trainIdx for kp in matches] #在训练图上的关键点的（kp2 ）匹配点索引

    post1 = np.int32([kp1[pp].pt for pp in p1]) #该点在vis上的坐标，落在img1上
    post2 = np.int32([kp2[pp].pt for pp in p2])+(w1,0) #该点在vis上的坐标，落在img2上，需要加上img1的宽度

    #绘制匹配线
    #zip:合并两列
    for (x1,y1),(x2,y2) in zip(post1,post2):
        cv2.line(vis,(x1,y1),(x2,y2),(0,0,255))
    #展示
    cv2.imshow("match",vis)
    cv2.waitKey()

img1_gray = cv2.imread("iphone1.png")
img2_gray = cv2.imread("iphone2.png")

#创建检测器 特征检测
sift = cv2.SIFT.create()
kp1,ds1 = sift.detectAndCompute(img1_gray,None)
kp2,ds2 = sift.detectAndCompute(img2_gray,None)

#匹配描述
#BFMatcher：基于L2范数的暴力匹配器
bf = cv2.BFMatcher(cv2.NORM_L2)
#使用 knnMatch 方法进行 KNN 匹配，每个描述符找两个最佳匹配。
matches = bf.knnMatch(ds1,ds2,k=2)

'''
matches = [
    [DMatch(queryIdx=0, trainIdx=5, distance=0.2), DMatch(queryIdx=0, trainIdx=10, distance=0.3)],
    [DMatch(queryIdx=1, trainIdx=3, distance=0.25), DMatch(queryIdx=1, trainIdx=8, distance=0.4)],
    ...
]
'''

goodMatches = []

'''
m 和 n 是两个 DMatch 对象，分别表示最近邻和次近邻的匹配结果。
m.distance 和 n.distance 分别是最近邻和次近邻匹配的距离，表示描述符之间的差异。
如果最近邻匹配的距离小于次近邻匹配距离的 50%，则认为这个匹配是好的，并将其添加到 goodMatch 列表中。
'''
for m,n in matches:

    if m.distance < 0.50 * n.distance:
        goodMatches.append(m)

#绘制匹配结果
drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatches)