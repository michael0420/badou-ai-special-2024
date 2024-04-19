import numpy as np
import cv2

"""
双线性插值法
"""

img = cv2.imread("lenna.png")
def bilinear_interpolation(img,out_img):  #bilinear_interpolation 双线性插值
    src_h,src_w,channel = img.shape         # 获取原图片数据
    dst_h,dst_w = out_img[1],out_img[0]     # 目标图片的 高和宽
    print("src_h，src_w原图的行高和宽：",src_h,src_w)     # 打印原图高和宽
    print("dst_h，dsr_w目标图的行高和宽：",dst_h,dst_w)   # 打印目标图高和宽
    if src_h == dst_h and src_w == dst_w:   # 如果原图和目标图高和宽一致
        return img.copy()
    dst_img =  np.zeros((dst_h,dst_w,3),dtype=np.uint8)  # 创建目标图片
    scale_x,scale_y = float(src_w) / dst_w,float(src_h) / dst_h # 计算缩放比例
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 目标q图x，y在原图中的位置（中心点重合）,src_x,src_y是目标图在原图中对应的点
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 找src_x，src_y 周边四个像素点的x0,x1,y0,y1
                src_x0 = int(np.floor(src_x))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1 ,src_w - 1)  # x1是x0右边的像素点，min限制x1的范围
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1 ,src_h -1)   # y1是y0下边的像素点，min限制y1的范围
                # 先做两个src_x,src_y x坐标方向的插值
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                # 在做src_x,src_y y坐标方向的插值,得到src_x,src_y的像素值，赋值给目标图
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y -src_y0) * temp1)
    return dst_img

if __name__ == "__main__":  # 主函数入口，被引用时，下面的代码不被运行
    img = cv2.imread("lenna.png")
    dst = bilinear_interpolation(img,(700,700))  # bilinear_interpolation 双线性插值
    cv2.imshow("bilinear_interpolation双线性插值：",dst)
    cv2.waitKey()