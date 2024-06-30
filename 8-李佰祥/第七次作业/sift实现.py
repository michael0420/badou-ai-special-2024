import cv2
import  numpy as np

img = cv2.imread("../../../img/dark2.jpg",0)

sift = cv2.SIFT_create()
keypoints , descriptor = sift.detectAndCompute(img,None)

img = cv2.drawKeypoints(image=img,outImage=img,
                  keypoints=keypoints,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51,163,236))

cv2.imshow('sift',img)
cv2.waitKey(0)
cv2.destroyAllWindows()












