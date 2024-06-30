import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

cv2.imshow("222",dst)
cv2.waitKey()

plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

cv2.imshow("Histogram equalization",np.hstack([gray,dst]))
cv2.waitKey()