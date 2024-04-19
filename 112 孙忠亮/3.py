import numpy as np
import cv2
#双线性插值
def bilinear_interpolation(img,out_dim):
    src_height, src_width ,channel= img.shape
    dst_height, dst_width = out_dim
    if src_height == dst_height and src_width == dst_width:
        return img.copy()
    dst_img=np.zeros((dst_height,dst_width,3),dtype=np.uint8)
    scale_x, scale_y = float(src_width)/float(dst_width),float(src_height)/float(dst_height)
    for i in range(channel):
        for dst_x in range(dst_height):
            for dst_y in range(dst_width):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_width - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_height - 1)
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img

img = cv2.imread('lenna.png')
image=bilinear_interpolation(img,(700,700))
cv2.imshow('image',image)
cv2.waitKey(0)

#直方图均衡化
img=cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)
dst=cv2.equalizeHist(img)
cv2.imshow('', np.hstack([img, dst]))
cv2.waitKey(0)

#sobel
img=cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)
x=cv2.Sobel(img,cv2.CV_16S,dx=1,dy=0)
y=cv2.Sobel(img,cv2.CV_16S,dx=0,dy=1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst=cv2.addWeighted(absX,0.5,absY,0.5,0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
