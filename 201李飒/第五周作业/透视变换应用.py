import cv2
import numpy as np


img = cv2.imread("photo1.jpg")
print(img.shape)
result3 = img.copy()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
m = cv2.getPerspectiveTransform(src,dst)
print(m)
result = cv2.warpPerspective(result3,m,(337,488))
cv2.imshow("src",img)
cv2.imshow("result",result)
cv2.waitKey()