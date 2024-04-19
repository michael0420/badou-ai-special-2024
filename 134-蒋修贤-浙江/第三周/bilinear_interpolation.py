#!/usr/bin/env python3
import numpy as np
import cv2

img = cv2.imread('lenna.jpeg')
h,w,c = img.shape
d_h = 700
d_w = 700
dst_img = np.zeros((d_h,d_w,c),dtype=np.uint8)

sc_x =float(w/d_w)
sc_y =float(h/d_h)

for i in range(c):
	for y in range(d_h):
		for x in range(d_w):
			src_x = (x+0.5)*sc_x-0.5
			src_y = (y+0.5)*sc_y-0.5
			
			src_x_l = int(np.floor(src_x))
			src_x_r = min(src_x_l+1,w-1)
			src_y_l = int(np.floor(src_y))
			src_y_r = min(src_y_l+1,h-1)
			
			l_num = (src_x_r - src_x) * img[src_y_l,src_x_l,i] + (src_x - src_x_l) * img[src_y_l,src_x_r,i]
			r_num = (src_x_r - src_x) * img[src_y_r,src_x_l,i] + (src_x - src_x_l) * img[src_y_r,src_x_r,i]
			
			dst_img[y,x,i]=int((src_y_r - src_y) * l_num + (src_y-src_y_l) * r_num)

cv2.imshow("dst", dst_img)
cv2.imshow("img", img)
cv2.waitKey()