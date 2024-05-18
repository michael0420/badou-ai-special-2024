import cv2
import numpy as np

img = cv2.imread('lenna.png',0)
cv2.imshow("canny",cv2.Canny(img,50,150))
cv2.waitKey(0)