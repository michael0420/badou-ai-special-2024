import cv2
import  numpy as np



def drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch):

    h1,w1=img1_gray.shape[:2]
    h2,w2=img2_gray.shape[:2]
    vis=np.zeros((max(h1,h2),w1+w2,3),np.uint8)
    vis[:h1,:w1,:]=img1_gray
    vis[:h2,w1:w1+w2,:]=img2_gray

    p1=[ kpp.queryIdx for kpp in goodMatch]
    p2=[ kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)

img1=cv2.imread("iphone1.png")
img2=cv2.imread("iphone2.png")

sift=cv2.SIFT_create()

kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)

bf=cv2.BFMatcher(cv2.NORM_L2)
matcher=bf.knnMatch(des1,des2,2)

Matches=[]
for m,n in matcher:
    if m.distance<0.5*n.distance:
        Matches.append(m)

drawMatchesKnn_cv2(img1,kp1,img2,kp2,Matches)

cv2.waitKey(0)
cv2.destroyAllWindows()