import cv2
import numpy as np

img = cv2.imread("Snipaste_2024-04-25_21-50-00.jpg")
print(img.shape)
cv2.imshow("img",img)
src = np.float32([[795, 290], [915, 260], [838, 386], [940, 360]])
dst = np.float32([[0, 0], [750, 0], [0, 750], [750, 750]])
img_zip = cv2.getPerspectiveTransform(src,dst)
result = cv2.warpPerspective(img,img_zip,(750,750))
cv2.imshow("img1",result)
cv2.waitKey(0)
