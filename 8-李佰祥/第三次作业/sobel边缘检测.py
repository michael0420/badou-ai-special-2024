import cv2
import numpy

img  =  cv2.imread("../../img/bird.png")


x = cv2.Sobel(img,cv2.CV_64F,1,0)
y = cv2.Sobel(img,cv2.CV_64F,0,1)

absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)


dst = cv2.addWeighted(absx,0.5,absy,0.5,0)


cv2.imshow("x", x)
cv2.imshow("y", y)

cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()






















