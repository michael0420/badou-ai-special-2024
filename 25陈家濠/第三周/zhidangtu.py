import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("lenna.png",1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst=cv2.equalizeHist(gray)

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
