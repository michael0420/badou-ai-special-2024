import numpy as np
import cv2
# sobel
img = cv2.imread("lenna.png",0)
x = cv2.Sobel(img,cv2.CV_16S, 1,0)
y = cv2.Sobel(img,cv2.CV_16S, 0,1)

absX = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX,0.5,absy,0.5,0)
cv2.imshow("absX", absX)
cv2.imshow("absY", absy)
cv2.imshow("result", dst)
cv2.waitKey()
cv2.destroyAllWindows()