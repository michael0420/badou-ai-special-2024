"""
@author: huangyunxuan

Canny调用接口

"""

import cv2


img = cv2.imread("lenna.png")
img_c = cv2.Canny(img,100 ,200)
cv2.imshow("img_c",img_c)
cv2.waitKey()