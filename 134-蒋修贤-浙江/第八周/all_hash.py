#!/usr/bin/env python3
import cv2
import numpy as np
import time
import os.path as path

#diff hash
def diffHash(img,width = 9,high = 8):
	img = cv2.resize(img,(width,high),interpolation=cv2.INTER_CUBIC)
	#gray
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	hash_str = ''
	sum = 0
	for i in range(high):
		for j in range(high):
			if j<7 and gray[i,j] >= gray[i,j+1]:
				hash_str += '1'
			else:
				hash_str += '0'
			sum += gray[i,j]
	print("diff hash")			
	print(hash_str)
	
	#avg
	avg = sum/(width*high)
	avg_hash = ''
	for i in range(width):
		for j in range(high):
			sum += gray[i,j]
			if gray[i,j] >= avg:
				avg_hash += '1'
			else:
				avg_hash += '0'
	print("avg hash")			
	print(avg_hash)
	
	return hash_str

if __name__=="__main__":
	img = cv2.imread('lenna.jpeg')
	
	diffHash(img, 8, 8)