import cv2
import matplotlib.pyplot as plt
import numpy as np

def func(h,w,img):
    high, width, channel = img.shape
    emptying = np.zeros((h,w,channel),np.uint8)
    nh = h/high
    nw = w/width
    for i in range(h):
        for j in range(w):
            x = int(i / nh + 0.5)
            y = int(j / nw + 0.5)
            emptying[i,j] = img[x,y]
    return emptying

img = cv2.imread('./lenna.png')
empty = func(500,500,img)
print(empty)
print(empty.shape)
print(img.shape)
cv2.imshow("nearest interp",empty)
cv2.imshow("image",img)
cv2.waitKey(0)





