import cv2
import numpy as np
img=cv2.imread("lenna.png")
height,width,channels=img.shape
biggerimage=np.zeros((1000,1000,channels),np.uint8)
bh=1000/height
bw=1000/width
for i in range(1000):
    for j in range(1000):
        x=int(i/bh+0.5)
        y=int(j/bw+0.5)
        biggerimage[i,j]=img[x,y]
cv2.imshow("nearest interp",biggerimage)
cv2.imshow("image",img)
cv2.waitKey(0)

