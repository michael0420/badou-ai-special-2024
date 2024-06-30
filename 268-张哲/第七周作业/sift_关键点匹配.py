import cv2
import numpy as np

def drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch):
    h1,w1 = img1_gray.shape[:2]
    h2,w2 = img2_gray.shape[:2]
    window = np.zeros((max(h1,h2),w1+w2,3),np.uint8)
    window[:h1,:w1] = img1_gray
    window[:h2,w1:w1+w2] = img2_gray
    post1 = np.int32([kp1[m.queryIdx].pt for m in goodMatch])
    post2 = np.int32([kp2[m.trainIdx].pt for m in goodMatch]) + (w1,0)
    for (x1,y1), (x2,y2) in zip(post1,post2):
        cv2.line(window,(x1,y1) ,(x2,y2), (0,0,255))
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", window)

img_g1 = cv2.imread('iphone1.png')
img_g2 = cv2.imread('iphone2.png')
sift = cv2.xfeatures2d.SIFT_create()
kp1,des1 = sift.detectAndCompute(img_g1,None)
kp2,des2 = sift.detectAndCompute(img_g2,None)
bf = cv2.BFMatcher(cv2.NORM_L2)
matchs = bf.knnMatch(des1,des2,k = 2)
goodmatch = []
for m,n in matchs:
    if m.distance < 0.5*n.distance:
        goodmatch.append(m)
drawMatchesKnn_cv2(img_g1,kp1,img_g2,kp2,goodmatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()