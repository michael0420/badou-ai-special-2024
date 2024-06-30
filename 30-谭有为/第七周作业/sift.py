import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取两张灰度图片
img1=cv2.imread('F:/PNG/iphone1.png')
gray1=cv2.cvtColor(img1,cv2.COLOR_BGRA2GRAY)
img2=cv2.imread('F:/PNG/iphone2.png')
gray2=cv2.cvtColor(img2,cv2.COLOR_BGRA2GRAY)

#提取关键点
sift=cv2.xfeatures2d.SIFT_create()  #声明函数
kp1,des1=sift.detectAndCompute(gray1,None)  #kp--关键点   des--描述符
kp2,des2=sift.detectAndCompute(gray2,None)
cv2.drawKeypoints(img1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51, 163, 236))
cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51, 163, 236))

#显示经过sift处理后的两张图片
plt.subplot(211)
plt.imshow(img1[:,:,::-1])  #bgr 转换 rgb
plt.title('sift')
plt.xticks([])
plt.yticks([])
plt.subplot(212)
plt.imshow(img2[:,:,::-1])
plt.xticks([])
plt.yticks([])
plt.show()

#特征点匹配
#opencv中knnMatch是一种蛮力匹配
#将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。
bf=cv2.BFMatcher(cv2.NORM_L2)  #函数声明
matchs=bf.knnMatch(des1,des2,k=2)  #使用欧式距离匹配特征点，每组匹配两个点m.n

#选择最佳匹配结果
goodMatch=[]
for m,n in matchs:
    if m.distance<0.5*n.distance:  #knnmatch每组匹配两个，如果第二个的距离比第一个的一半还少，认为是一个好的匹配
        goodMatch.append(m)
print(goodMatch)

#使用cv2.drawMacthesKnn连接两张图片的匹配点
sift_img=cv2.drawMatches(img1,kp1,img2,kp2,goodMatch,None,flags=2)
cv2.imshow('sift_img',sift_img)
cv2.waitKey()




