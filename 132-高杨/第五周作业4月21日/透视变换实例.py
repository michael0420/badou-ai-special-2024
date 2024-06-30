import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
res =img.copy()


src = np.float32([[207,151],[517,285],[17,601],[343,731]])
dst = np.float32([[0,0],[337,0],[0,488],[337,488]])


m = cv2.getPerspectiveTransform(src,dst)
print(f"warpMatrix:  {m} ")

res = cv2.warpPerspective(res,m,(337,488))
cv2.imshow('orgin',img)
cv2.imshow('result',res)

cv2.waitKey(0)