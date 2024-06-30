"""
@author: huangyunxuan
Sobel

"""
import cv2

img = cv2.imread("lenna.png")

x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)

abs_x = cv2.convertScaleAbs(x)
abs_y = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(abs_x,0.5,abs_y,0.5,0)
cv2.imshow("dst",dst)
cv2.imshow("img",img)
cv2.waitKey(0)


