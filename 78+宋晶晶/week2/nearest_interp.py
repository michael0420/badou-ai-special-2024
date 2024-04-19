import cv2
import os
import numpy as np

HERE = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(HERE)
FILE_PATH = os.path.join(PARENT_DIR,'src','lenna.png')

def nearest(img):
    h,w,channels = img.shape
    emptyImg = np.zeros((800,800,channels),img.dtype)
    sh = 800/h
    sw = 800/w
    for i in range(800):
        for j in range(800):
            x = int(i/sh+0.5) #向下取整
            y = int(j/sw + 0.5)
            emptyImg[i,j] = img[x,y]
    
    return emptyImg

img = cv2.imread(FILE_PATH)
zoom = nearest(img)

cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)