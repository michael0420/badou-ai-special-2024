import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("1947.jpg", 1)
cv2.imshow("img", img)
(b, g, r) = cv2.split(img)

hb = cv2.equalizeHist(b)
hr = cv2.equalizeHist(r)
hg = cv2.equalizeHist(g)

cv2.imshow("hb", hb)
cv2.imshow("hr", hr)
cv2.imshow("hg", hg)

dst = cv2.merge((hb, hr, hg))

cv2.imshow("dst", dst)

cv2.waitKey()

