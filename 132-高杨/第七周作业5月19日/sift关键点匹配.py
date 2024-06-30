import cv2
import numpy as np


def ShowMatchResult(img_gray1,img_gray2,keypoint1,keypoint2,goodMatch):
    h1,w1=img_gray1.shape[:2]
    h2,w2=img_gray2.shape[:2]

    MatchPic = np.zeros((max(h1,h2),w1+w2,3),np.uint8)

    MatchPic[:h1,:w1]=img_gray1
    MatchPic[:h2,w1:w1+w2]=img_gray2


    # 在goodmatch 中寻找已经匹配好的点
    pp_1=[knn.queryIdx  for knn in goodMatch]
    pp_2=[knn.trainIdx  for knn in goodMatch]

    # 获取每个图像的关键点坐标
    post_1 = np.int32([  keypoint1[pp].pt  for pp in pp_1])
    #后面要把两张图显示在一张上，所以将第二张图特征点+左边图宽度整体右移到右边 这是w1的作用
    post_2 = np.int32([  keypoint2[pp].pt  for pp in pp_2]) + (w1,0)

    # 画图

    for (x1,y1),(x2,y2) in zip(post_1,post_2):
        cv2.line(MatchPic,(x1,y1),(x2,y2),color=(255,255,0))

    cv2.namedWindow('MatchResult',cv2.WINDOW_NORMAL)
    cv2.imshow('MatchResult',MatchPic)

img_1 = cv2.imread('iphone1.png')
img_2 = cv2.imread('iphone2.png')




sift = cv2.SIFT_create()

keypoint1,descriptor1=sift.detectAndCompute(img_1,None)
keypoint2,descriptor2=sift.detectAndCompute(img_2,None)


# Match工具 ， 用欧氏距离来设置match方法
bf = cv2.BFMatcher(cv2.NORM_L2)
#用描述子来确定1和2的匹配程度
matches = bf.knnMatch(descriptor1,descriptor2,k=2)

goodmatch=[]
for i,j in matches:
    # k=2 比较 两个点
    if i.distance<0.50*j.distance:
        goodmatch.append(i)

ShowMatchResult(img_gray1=img_1,img_gray2=img_2,keypoint1=keypoint1,keypoint2=keypoint2,goodMatch=goodmatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()