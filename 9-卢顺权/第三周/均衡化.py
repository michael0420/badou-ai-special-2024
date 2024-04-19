import  cv2
import matplotlib.pyplot as plt
img=cv2.imread('LYF.jpeg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.equalizeHist(img_gray)#直方图均衡化的方法
hist = cv2.calcHist([img2],[0],None,[256],[0,256])
plt.figure()
plt.hist(img2.ravel(),256)
plt.show()
cv2.waitKey()