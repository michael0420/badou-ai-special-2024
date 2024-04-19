import cv2

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(gray)

cv2.imshow('gary1', gray)
cv2.imshow('gray2', dst)
cv2.waitKey()
cv2.destroyAllWindows()
