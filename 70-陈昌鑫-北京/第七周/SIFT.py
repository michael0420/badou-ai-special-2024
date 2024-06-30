import sys
from cv2 import xfeatures2d
import cv2
import numpy as np

def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
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

if len(sys.argv) >= 3:
  im1f, im2f = sys.argv[1], sys.argv[2]
else:
  im1f = 'library1.jpg'
  im2f = 'library2.jpg'

img1_gray = cv2.imread(im1f)
img2_gray = cv2.imread(im2f)
# sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.SURF()

kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

bf = cv2.BFMatcher(cv2.NORM_L2)

matches = bf.knnMatch(des1, des2, k = 2)
goodMatch = []
for m,n in matches:
    if m.distance < 0.50*n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch[:50])

cv2.waitKey(0)
cv2.destroyAllWindows()