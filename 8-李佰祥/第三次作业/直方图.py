import cv2
import numpy
import matplotlib.pyplot as plt

img = cv2.imread("../../lenna.png", 0)
cv2.imshow("img",img)

#实现均衡化
dst = cv2.equalizeHist(img)

hist = cv2.calcHist([dst],[0],None,[256],[0,256])

cv2.imshow("dst",dst)










#
cv2.waitKey(0)
cv2.destroyAllWindows()


















