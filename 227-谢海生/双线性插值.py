import numpy as np
import cv2

#定义名为bilinear_interpolation的函数，接受两个参数img,out_dim
def bilinear_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w",src_h, src_w)
    print ("dst_h, dst_w",dst_h, dst_w)
#如果输入原图的高和宽等于目标图像的高和宽则返回，继续执行代码
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

#初始化名为dst_img的空numpy的数组,数据类型为无符号8位整数
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
#计算原图与目标图的缩放比例
    scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
#使用几何中心对称方法计算了从目标图像找原图坐标的方法，因为都是虚拟坐标点，离得最近的影响越大，同时这个通过公式代入得到0.5，缩放比例-0.5其实是
#原图像坐标+0.5左右移项
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

#使用# np.floor() 函数向下取整，将 src_x 和 src_y 变量分别四舍五入到最接近的整数，得到 src_x0 和 src_y0。
#代码计算了 src_x1 和 src_y1 的值，它们是 src_x0 + 1 和 src_y0 + 1 的最小值，分别小于或等于 src_w - 1 和 src_h - 1
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1,src_h - 1)
#使用双线性插值计算，公式就是ppt中的方法计算公式
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img
#惯用语，运行主模块
if __name__ == '__main__':
    #读图
    img = cv2.imread("C:/Users/86188/Pictures/lenna.png")
    dst = bilinear_interpolation(img,(700,700))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()










