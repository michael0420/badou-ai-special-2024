import cv2
import numpy as np

from skimage.color import rgb2gray


def gray_demo(img):
    if img is None:
        print("load picture error!")
    else:
        h,w = img.shape[:2]
        img_gray = np.zeros([h,w],img.dtype)
        for i in range(h):
            for j in range(w):
                m = img[i,j]
                img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
    cv2.imshow("image gray",img_gray)




def binary_demo(img):
    if img is None:
        print("load picture error!")
    else:
        img_gray = rgb2gray(img)
        row,col = img_gray.shape
        for i in range(row):
            for j in range(col):
                if(img_gray[i,j] <= 0.5):
                    img_gray[i,j] = 0
                else:
                    img_gray[i,j] = 1
    cv2.imshow("binary picture",img_gray)


img = cv2.imread("lenna.png",cv2.IMREAD_COLOR)

gray_demo(img)


binary_demo(img)
# cv2.imshow("demo",img_gray)



cv2.imshow("lenna",img)
cv2.waitKey(0)
# print(img)