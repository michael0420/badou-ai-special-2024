import cv2
import numpy as np


#SIFT算法--找出关键点
def fun1():
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints,descriptor = sift.detectAndCompute(gray,None)

    #对得到的关键点绘制圆圈和方向
    img = cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            clor=(51,163,236))

    cv2.imshow('sift_keypoints',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#特征匹配
def fun2():
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)

img1_gray = cv2.imread("iphone1.png")
img2_gray = cv2.imread("iphone2.png")

sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(img1_gray,None)
kp2,des2 = sift.detectAndCompute(img2_gray,None)

#通过欧氏距离对两张图匹配
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1,des2,k = 2)  #找出相似度最高的前2个

goodMatch = []
for m,n in matches:
    if m.distance < 0.50*n.distance:
        goodMatch.append(m)

fun2(img1_gray,kp1,img2_gray,kp2,goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()


if __name__ == '__main__':
    fun1()
    #fun2()
