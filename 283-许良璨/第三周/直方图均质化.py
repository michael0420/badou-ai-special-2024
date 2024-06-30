import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread("../lenna.png")
gray1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

dst=cv2.equalizeHist(gray1)
hist=cv2.calcHist([dst],[0],None,[256],[0,256])

cv2.imshow("OUTCOME：",np.hstack([gray1,dst]))
plt.figure()
plt.hist(dst.ravel(), 256)
plt.title('after equalization')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()