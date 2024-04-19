import cv2

img = cv2.imread("lenna.png")
res = cv2.merge(tuple([cv2.equalizeHist(x) for x in cv2.split(img)]))

cv2.imshow("dst_rgb", res)
cv2.waitKey(0)