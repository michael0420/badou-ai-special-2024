import cv2

img = cv2.imread('lenna.png')
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

rx = cv2.convertScaleAbs(x)
ry = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(rx, 0.5, ry, 0.5, 0)
cv2.imshow('result', dst)
cv2.imshow('result2', rx)
cv2.imshow('result3', ry)
cv2.waitKey(0)
cv2.destroyAllWindows()