import cv2
import numpy as np

#双线性插值实现
def bilinear_interpolation(img,out_size):
    src_h,src_w,channel = img.shape
    dst_h,dst_w = out_size[1],out_size[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x , scale_y = float(src_w)/dst_w , float(src_h)/dst_h
    for i in range(channel):
        for y in range(dst_h):
            for x in range(dst_w):
                '''
                 求目标图像的对应原图的src_x 和 src_y 坐标
                 使用几何中心对称
                 如果使用直线方式， src_x = dst_x * scale_x
                 将两幅图像中心对称 (x+0.5)*scale_x -0.5
                '''
                src_x = (x+0.5)*scale_x -0.5
                src_y = (y+0.5)*scale_y -0.5
                # 计算差值点的坐标x0,y0,x1,y1
                src_x0 = int(np.floor(src_x)) #np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0+1,src_w-1)
                src_y1 = min(src_y0+1,src_h-1)
                # 计算插值
                temp0 = (src_x1-src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1-src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[y,x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png',1)
    dst = bilinear_interpolation(img,(700,700))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey(0)

