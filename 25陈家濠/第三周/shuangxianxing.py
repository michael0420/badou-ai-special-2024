import numpy as np
import cv2
def bilinear_interpolation(img,out_size):
    src_h,src_w,channal=img.shape
    dst_h,dst_w=out_size[0],out_size[1]
    if src_h==dst_h and src_w==dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3),dtype=np.uint8)
    x,y=float(src_w)/dst_w,float(src_h)/dst_h
    for i in range(channal):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x=(dst_x+0.5)*x-0.5
                src_y=(dst_y+0.5)*y-0.5
                src_x1=int(np.floor(src_x))
                src_x2=min(src_x1+1,src_w-1)
                src_y1=int(np.floor(src_y))
                src_y2=min(src_y1+1,src_w-1)
                temp0=(src_x2-src_x)*(img[src_y1,src_x1,i])+(src_x-src_x1)*(img[src_y1,src_x2,i])
                temp1= (src_x2 - src_x) * (img[src_y2, src_x1, i]) + (src_x - src_x1) * (img[src_y2, src_x2, i])
                dst_img[dst_y,dst_x,i]=int((src_y2-src_y)*temp0+(src_y-src_y1)*temp1)
    return dst_img

img = cv2.imread('lenna.png')
dst = bilinear_interpolation(img,(700,700))
cv2.imshow('bilinear interp',dst)
cv2.waitKey()
