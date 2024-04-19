import cv2

img_gray = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)

img_he = cv2.equalizeHist(img_gray)

cv2.imshow("source", img_gray)
cv2.imshow("dest", img_he)
cv2.waitKey(0)
