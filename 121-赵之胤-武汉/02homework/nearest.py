import cv2
import numpy


def nearest(img):
    h, w, c =img.shape
    neatest_img = numpy.zeros((1000, 1000, c), numpy.uint8)
    h_factor = 1000/h
    w_factor = 1000/w
    for i in range(1000):
        for j in range(1000):
            x = int(i/h_factor + 0.5)
            y = int(j/w_factor + 0.5)
            neatest_img[i, j] = img[x, y]
    return neatest_img


img = cv2.imread("lenna.png")
result = nearest(img)
cv2.imshow("img", img)
cv2.imshow("nearest", result)
cv2.waitKey(0)
