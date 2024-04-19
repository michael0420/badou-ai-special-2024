
import numpy as np
import matplotlib.pyplot as plt
import cv2

def img2gray(img):
    height, width = img.shape[:2]
    img_gray = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            img_gray[i, j] = int(img[i, j, 0] * 0.11 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.3)
    return img_gray

def img2binary(img):
    img_binary = np.where(img_gray >= 128, 255, 0)
    return img_binary

def img_nearest_interp(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh + 0.5)
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage

img = cv2.imread("lenna.png")
img_gray = img2binary(img)
img_binary = img2binary(img)
zoom = img_nearest_interp(img)

cv2.imshow("image",img)
cv2.imshow("gray image",img_gray)
cv2.imshow("binary image",img_binary)
cv2.imshow("nearest interp",zoom)
cv2.waitKey(0)
