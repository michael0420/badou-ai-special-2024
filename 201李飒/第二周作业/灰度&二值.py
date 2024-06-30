import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
h,w = img.shape[:2]
grey_img = np.zeros((h,w,3),img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        grey_img[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
cv2.imshow("GREY",grey_img)
cv2.waitKey()

img1 = plt.imread("lenna.png")
img_gray = rgb2gray(img1)
plt.imshow(img_gray,cmap="gray")
plt.show()

rows,cols = img_gray.shape
img_binary = np.zeros((rows,cols,1),img_gray.dtype)
for i in range(rows):
    for j in range(cols):
         if (img_gray[i,j]>0.5):
             img_binary[i,j]=1
        else:
           img_binary[i,j] =0

plt.imshow(img_binary,cmap="gray")
plt.show()



