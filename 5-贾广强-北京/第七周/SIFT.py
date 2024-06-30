import cv2
import numpy as np

img = cv2.imread('lenna.jpg')


sift = cv2.SIFT_create()
keypoints,descriptor = sift.detectAndCompute(img,None)
img1 = cv2.drawKeypoints(image=img,outImage=None,keypoints= keypoints,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                         color=(51,163,236))
cv2.imshow('h',np.hstack([img1,img]))

cv2.waitKey()
cv2.destroyAllWindows()



