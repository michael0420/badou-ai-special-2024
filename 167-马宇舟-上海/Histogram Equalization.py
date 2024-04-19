import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\lenovo\Desktop\lenna.png", 1)


(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

result = cv2.merge((bH, gH, rH))
cv2.imshow('dst_image', result)
cv2.imshow('src', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst], [0], None, [256], [0,256])
plt.figure()
plt.plot(hist, Color = 'Blue')
plt.xlim([0, 256])
# plt.hist(dst.ravel(), 256)
plt.show()


cv2.imshow('Histogram Equalization', np.hstack([gray, dst]))
cv2.waitKey(0)
