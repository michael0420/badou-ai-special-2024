import cv2
import numpy as np

img = cv2.imread('maruko.png')

result = img.copy()

src = np.float32([[183,379],[902,155],[512,1362],[1226,1095]])
dst = np.float32([[0,0],[400,0],[0,600],[400,600]])
print(img.shape)

M = cv2.getPerspectiveTransform(src,dst)
print('warpmnatrix:',M)
#print(tou)

result2 = cv2.warpPerspective(result,M,(400,600))
cv2.imshow('src',img)
cv2.imshow('result',result2)
cv2.waitKey()