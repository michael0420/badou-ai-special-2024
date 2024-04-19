import cv2
import numpy as np


img = cv2.imread("lenna.png")
h,w,c = img.shape
newh = 800
neww = 800
newimg = np.zeros((newh, neww, c), np.uint8)
sh = newh/h
sw = neww/w
for i in range(newh):
    for j in range(neww):
        x = int(i/sh + 0.5)
        y = int(j/sw + 0.5)
        newimg[i, j] = img[x, y]
        #print(sh,sw,i,j,x,y)
cv2.imshow("oldimg", img)
cv2.imshow("image", newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
