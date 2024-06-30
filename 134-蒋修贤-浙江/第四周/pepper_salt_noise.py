#!/usr/bin/env python3
import numpy as np
import cv2
from numpy import shape
import random

img = cv2.imread('lenna.jpeg',0)

noice_num = int(0.3 *img.shape[0]*img.shape[1])

for i in range(noice_num):
	rX = random.randint(0, img.shape[0]-1)
	rY = random.randint(0, img.shape[1]-1)
	if random.random() < 0.5:
		img[rX,rY] = 0
	else:
		img[rX,rY] = 255
		
cv2.imshow('pepper salt',img)
cv2.waitKey()