
import numpy as np
import cv2
 

def bilinear_interpolation(img,out_dim):
	
    src_h, src_w, channel = img.shape
    #获取输入图像的高度、宽度和通道数,分别赋值给变量 src_h,src_w 和 channel
    dst_h, dst_w = out_dim[1], out_dim[0]
    #从输出尺寸元组 out_dim 中获取目标图像的高度和宽度,分别赋值给变量 dst_h 和 dst_w
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
    	#如果输入图像的尺寸与目标图像的尺寸相同,直接返回输入图像的副本
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    #创建一个形状为 (dst_h, dst_w, 3) 的全零数组,数据类型为无符号 8 位整数,用于存储目标图像
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    #计算宽度和高度方向上的缩放比例,分别赋值给变量 scale_x 和 scale_y
    for i in range(channel):
    	#对每个颜色通道进行循环处理
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
            #对目标图像的每个像素位置 (dst_x, dst_y) 进行循环处理
 
                
                src_x = (dst_x + 0.5) * scale_x-0.5
                src_y = (dst_y + 0.5) * scale_y-0.5
                #计算目标图像坐标 (dst_x, dst_y) 在原图像中对应的浮点坐标 (src_x, src_y)
 
                
                src_x0 = int(np.floor(src_x))     #np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1 ,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                #计算 (src_x, src_y) 周围的四个整数坐标 
 
                
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                #分别计算 (src_x, src_y) 在 y 方向上的两个临近点的插值结果,赋值给变量 temp0 和 temp1
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
                #计算 (src_x, src_y) 在 x 方向上的最终插值结果,并将结果赋值给目标图像的对应位置 dst_img[dst_y,dst_x,i]

 
    return dst_img
 
 
if __name__ == '__main__':
#主程序入口,只有当该文件作为主程序运行时才会执行以下代码。
    img = cv2.imread('lenna.png')
    #使用 OpenCV 的 imread 函数读取名为 "lenna.png" 的图像文件
    dst = bilinear_interpolation(img,(700,700))
    #调用 bilinear_interpolation 函数对图像进行双线性插值,目标图像尺寸为 (700, 700)
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()
