import cv2
import numpy as np

def nearInter(img,row,col):
    h,w,t = img.shape
    tmpImage = np.zeros([row,col,t],dtype=img.dtype)
    rh = row / h
    rw = col / w
    for r in range(row):
        for c in range(col):
            i = int( r / rh + 0.5)
            j = int( c / rw + 0.5)
            tmpImage[r,c] = img[i,j]
    return tmpImage

img = cv2.imread("lenna.png")
row = 800
col = 800
interpolationImg = nearInter(img,row,col)
cv2.imshow("near interp",interpolationImg)
cv2.imshow("imgage",img)
cv2.waitKey(0)