import cv2
import numpy as np


def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

img = cv2.imread("./test_1088x1735.jpg", 0)
img_sobel = sobel(img)
cv2.imshow("Result", img_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()
