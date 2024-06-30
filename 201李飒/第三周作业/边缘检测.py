import cv2

img = cv2.imread("lenna.png",0)

x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)

absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absx,0.5,absy,0.5,0)
cv2.imshow("X",absx)
cv2.imshow("Y",absy)
cv2.imshow("result",dst)
cv2.waitKey()
cv2.destroyAllWindows()
