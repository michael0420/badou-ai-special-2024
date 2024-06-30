import cv2

src_img = cv2.imread("lenna.png")


def hist_img(img):
    (b, g, r) = cv2.split(img)
    dst_b = cv2.equalizeHist(b)
    dst_g = cv2.equalizeHist(g)
    dst_r = cv2.equalizeHist(r)
    return cv2.merge((dst_b, dst_g, dst_r))


dst_img = hist_img(src_img)

cv2.imshow("src", src_img)
cv2.imshow("dst_img", dst_img)
cv2.waitKey(0)
