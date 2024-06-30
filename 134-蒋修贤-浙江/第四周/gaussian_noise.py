#!/usr/bin/env python3
import numpy as np
import cv2
from numpy import shape
import random

img = cv2.imread('lenna.jpeg',0)

noice_num = int(0.8*img.shape[0]*img.shape[1])

for i in range(noice_num):
	rX = random.randint(0, img.shape[0]-1)
	rY = random.randint(0, img.shape[1]-1)
	##gnerate gaussian num
	img[rX,rY] = img[rX,rY] + random.gauss(2,4)
	if img[rX,rY] < 0:
		img[rX,rY] = 0
	elif img[rX,rY] > 255:
		img[rX,rY] = 255

cv2.imshow('gaussian',img)
cv2.waitKey()