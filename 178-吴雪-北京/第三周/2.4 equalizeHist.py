"""
直方图均衡化 histogram equalization
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("E:/Desktop/jianli/lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalization = cv2.equalizeHist(gray)
histg = cv2.calcHist([gray], [0], None, [256], [0, 256])
histe = cv2.calcHist([equalization], [0], None, [256], [0, 256])
plt.subplot(221), plt.plot(histg, color='g')
plt.subplot(222), plt.plot(histe, color='g')
plt.subplot(223), plt.hist(gray.ravel(), 256)
plt.subplot(224), plt.hist(equalization.ravel(), 256)
plt.show()
cv2.imshow("histogram equalization", np.hstack([gray, equalization]))
cv2.waitKey(0)
cv2.destroyAllWindows()
