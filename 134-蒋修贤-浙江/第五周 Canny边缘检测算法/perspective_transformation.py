#!/usr/bin/env python3
import cv2
import numpy as np

img = cv2.imread('photo.jpg')

result = img.copy()

src = np.float32([[1047,556],[4677,654],[1435,3103],[4677,3032]])
dst = np.float32([[0,0],[5472,0],[0,3648],[5472,3648]])
m = cv2.getPerspectiveTransform(src, dst)
resultCv2 = cv2.warpPerspective(result, m, (5472,3648))
cv2.imshow('src', img)
cv2.imshow('result', resultCv2)
cv2.waitKey(0)
cv2.destroyAllWindows()