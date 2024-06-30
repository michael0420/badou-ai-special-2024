import cv2

SrcImg = cv2.imread('canny.png')
cv2.imshow('SrcImg', SrcImg)
img = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_img', img)
cv2.imshow('canny', cv2.Canny(img, 100, 445))
cv2.waitKey(0)
