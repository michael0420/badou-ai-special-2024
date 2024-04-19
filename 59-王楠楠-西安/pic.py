from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread("my.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        img_new = img[i,j]
        img_gray[i,j] = int(img_new[0]*0.11 + img_new[1]*0.59 + img_new[2]*0.3)

print (img_new)
print (img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)
cv2.imwrite('output.jpg', img_gray)
