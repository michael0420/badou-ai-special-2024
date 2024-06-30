import cv2


if __name__ == '__main__':
    img = cv2.imread("../data/lenna.png")

    b, g, r = cv2.split(img)

    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    equalized_image = cv2.merge((b_eq, g_eq, r_eq))
    cv2.imshow("lenna", img)
    cv2.imshow('luna', equalized_image)
    cv2.waitKey(0)

