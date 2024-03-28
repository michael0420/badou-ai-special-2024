import numpy as np
import cv2

def nearest_interpolation(src_img):
    des_img = np.zeros((des_h, des_w, src_img.shape[2]), np.uint8)
    for i in range(des_img.shape[0]):
        for j in range(des_img.shape[1]):
            x = int(i / des_img.shape[0] * src_img.shape[0] + 0.5)
            y = int(j / des_img.shape[1] * src_img.shape[1] + 0.5)
            x = x - 1 if x == src_img.shape[0] else x
            y = y - 1 if y == src_img.shape[1] else y
            des_img[i,j] = src_img[x,y]
    return des_img
src_img = cv2.imread("lenna.png")
des_h = int(input("pls input pic height:"))
des_w = int(input("pls input pic weight:"))

des_img = nearest_interpolation(src_img)
cv2.imshow("lenna", src_img)
cv2.imshow("nearest_interpolation", des_img)
cv2.waitKey(0)
