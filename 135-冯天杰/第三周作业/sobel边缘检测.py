import cv2

img = cv2.imread("69e754ae5b5a3ecf8f150894ee6b095.jpg", 0)

x = cv2.Sobel(img, cv2.CV_16S, 0, 1)
y = cv2.Sobel(img, cv2.CV_16S, 1, 0)

x1 = cv2.convertScaleAbs(x)
y1 = cv2.convertScaleAbs(y)

xy = cv2.addWeighted(x1, 0.5, y1, 0.5, 0)

cv2.imshow("x1", x1)
cv2.imshow("y1", y1)
cv2.imshow("xy", xy)
cv2.waitKey(0)
cv2.destroyAllWindows()
