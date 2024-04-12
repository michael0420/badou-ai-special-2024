import  cv2
import numpy as np
from matplotlib import  pyplot as plt

img=cv2.imread("lenna.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(0)

dst=cv2.equalizeHist(gray)
hst=cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

cv2.imshow("gray22",np.hstack([gray,dst]))
cv2.waitKey()
