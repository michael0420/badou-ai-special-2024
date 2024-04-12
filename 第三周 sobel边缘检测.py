import cv2
import numpy

rawImg = cv2.imread("/Users/mac/Desktop/tuanzi.jpg")

x = cv2.Sobel(rawImg,cv2.CV_16S,1,0)  #使用sobel对x轴进行边缘检测

y = cv2.Sobel(rawImg,cv2.CV_16S,0,1)  #使用sobel对y轴进行边缘检测

#因为原图格式是uint8，但sobel后值会超出uint8的0~255,所以需要增加位数
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

#将x和y方向sobel后的结果合并
targetImg = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)
cv2.imshow("Result", targetImg)

cv2.waitKey(0)
#cv2.destroyAllWindows()