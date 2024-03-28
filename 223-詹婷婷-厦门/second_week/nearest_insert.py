import cv2
import numpy as np

def function(img, h, w):
    h0, w0, c0 = img.shape
    img_insert = np.zeros([h,w,c0], np.uint8)
    for i in range(h):
        for j in range(w):
            img_insert[i,j] = img[int(i / (h / h0) + 0.5 ), int(j / (h / h0) + 0.5)]

    return img_insert


img = cv2.imread("lenna.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("original", img)

img_insert = function(img, 1023, 1023)
cv2.imshow("nearest insert", img_insert)


cv2.waitKey(0)