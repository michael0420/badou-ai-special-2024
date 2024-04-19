import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("lenna.png", 0)
dst = cv2.equalizeHist(img)

hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram", np.hstack([img, dst]))
cv2.waitKey(0)

img = cv2.imread("lenna.png", 1)

(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

result = cv2.merge((bH, gH, rH))
cv2.imshow("Histogram_rgb", np.hstack([img, result]))

cv2.waitKey(0)