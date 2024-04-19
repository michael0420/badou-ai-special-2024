import cv2 as cv
import numpy as np

img = cv.imread("lenna.png", 0)

#用有符号16位接住无符号8位
x = cv.Sobel(img, cv.CV_16S, 1, 0)
y = cv.Sobel(img, cv.CV_16S, 0, 1)

#获取绝对值，在0-255之间
abxx = cv.convertScaleAbs(x)
abxy = cv.convertScaleAbs(y)
cv.imshow("test-x", abxx)
cv.imshow("test -y", abxy)
newImg = cv.addWeighted(abxx, 0.5, abxy, 0.5, 0)
cv.imshow("test", newImg)
cv.waitKey()
cv.destroyAllWindows()