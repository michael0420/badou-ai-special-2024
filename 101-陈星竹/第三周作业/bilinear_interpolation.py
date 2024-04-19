import numpy as np
import cv2

def function(img,size_w,size_h):
    width,height,channels = img.shape
    scale_x = float(width)/size_w
    scale_y = float(height)/size_h

    if width== size_w and height == size_h:
        return img

    dst_img = np.zeros((size_h,size_w,3),dtype=np.uint8)

    for channel in range(channels):
        for w in range(size_w):
            for h in range(size_h):
                #几何中心对齐
                src_x = (w+0.5) * scale_x - 0.5
                src_y = (h+0.5) * scale_y -0.5

                #得到点(x,y)
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 +1,width-1) #边缘检测
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1,height-1)
                #得到4个相邻点 (x1,y0) (x1,y1) (x0,y1) (x0,y0)

                #矩阵[行（h）,列（w）]
                # ((x1-x)/x1-x0 )*f(x0,y0) + ((x-x0)/x1-x0) * f(x1,y0)
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,channel] + (src_x - src_x0) * img[src_y0,src_x1,channel]
                #((x1-x)/x1-x0) * f(x0,y1) + ((x-x0)/x1-x0) * f(x1,y1)
                temp1 = (src_x1-src_x) * img[src_y1,src_x0,channel] + (src_x - src_x0) * img[src_y1,src_x1,channel]
                #(y1-y)*temp0 + (y-y0)*temp1
                dst_img[h,w,channel] = int((src_y1-src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

img = cv2.imread('lenna.png')

if __name__ == '__main__':
    dst_img = function(img, 700, 700)
    cv2.imshow('bilinear interp', dst_img)
    cv2.waitKey()