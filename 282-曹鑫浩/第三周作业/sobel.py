import cv2

img = cv2.imread('lenna.png',0)

gradient_x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
gradient_y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

gradient_X = cv2.convertScaleAbs(gradient_x)
gradient_Y = cv2.convertScaleAbs(gradient_y)

img_dst = cv2.addWeighted(gradient_X, 0.5, gradient_Y, 0.5, 0)

cv2.imshow('Sobel', img_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()