import cv2
import numpy as np

def Cannythreshold(lowthreshold):
    detected_edges = cv2.Canny(gray, lowthreshold, lowthreshold*ratio, apertureSize = kernel_size)
    dst = cv2.bitwise_and(gray, gray, mask = detected_edges)
    cv2.imshow('canny result', dst)

lowthreshold = 0
max_lowthreshold = 100
ratio = 3
kernel_size = 3

img =cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny result')

cv2.createTrackbar('Min threshold','canny result',lowthreshold,max_lowthreshold,Cannythreshold)

Cannythreshold(0)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()