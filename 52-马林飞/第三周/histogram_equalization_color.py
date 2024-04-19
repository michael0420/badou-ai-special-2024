import cv2

img = cv2.imread("lenna.png")

(g, b, r) = cv2.split(img)

g_hist = cv2.equalizeHist(g)
b_hist = cv2.equalizeHist(b)
r_hist = cv2.equalizeHist(r)


img_dest = cv2.merge((g_hist, b_hist, r_hist))
cv2.imshow("img", img_dest)
cv2.waitKey(0)
