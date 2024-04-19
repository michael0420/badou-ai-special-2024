import time

import numpy as np
import cv2



def Color2Gray(image,gray_img,binary_img):
    for i in range(height):
        for j in range(width):
            b,g,r = img[i,j]
            gray_img[i, j] = int(b * 0.11 + g * 0.59 + r * 0.3)
            if gray_img[i, j]>100:
                binary_img[i, j] = 255
            else:
                binary_img[i, j] = 0

def NearestInterpolation(img,resize_height,resize_width):
    h_ori,w_ori,c_ori = img.shape
    emptyImage =  np.zeros([resize_height,resize_width,c_ori],np.uint8)
    resize_ratio_w = w_ori/resize_width
    resize_ratio_h = h_ori/resize_height

    for i in range(resize_height):
        for j in range(resize_width):
            y = int(resize_ratio_h * i - 0.5)
            x = int(resize_ratio_w * j - 0.5)

            emptyImage[i,j] = img[y,x]

    return emptyImage

img = cv2.imread("../../lenna.png")
height,width,C = img.shape
gray_img = np.zeros([height, width], img.dtype)
binary_img = np.zeros([height, width], img.dtype)
Color2Gray(img,gray_img,binary_img)
# cv2.imshow("src_image",img)
# cv2.imshow("gray_img",gray_img)
# cv2.imshow("binary_img",binary_img)
# cv2.waitKey(0)
resize_image = NearestInterpolation(img,50,50)
cv2.imshow("resize_image",resize_image)
cv2.waitKey(0)