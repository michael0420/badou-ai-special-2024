import cv2
import matplotlib.pyplot as plt


img = cv2.imread('lenna.png',1)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 直方图
hist = cv2.calcHist(gray,[0],None,[256],[0,256])
print(hist)

plt.figure()
plt.hist(hist,256)
plt.show()


#灰度图的直方图均衡化

target = cv2.equalizeHist(gray)

cv2.imshow('orgin',gray)

cv2.imshow('result',target)
cv2.waitKey(0)


