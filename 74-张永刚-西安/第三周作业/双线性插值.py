"""
@author: 74+张永刚
实现双线性插值
"""
import cv2
import numpy as np

def bilinear_interpolation (img,out_dim):
    # 获取原图像的大小及通道
    old_hide,old_wide,channel = img.shape
    # 新图像的大小
    new_hide,new_wide = out_dim[1],out_dim[0]

    # 判断新图像和原图像大小一样的话就直接返回
    if new_hide == old_hide and new_wide == old_wide:
        return img.copy()

    # 创建一个空图像
    new_img = np.zeros((new_hide,new_wide,3),dtype=np.uint8)
    # 计算出新老图像的比例关系
    scale_x,scale_y = float(old_wide) / new_wide, float(old_hide) / new_hide
    for i in range(channel):
        for new_y in range(new_hide):
            for new_x in range(new_wide):
                # 做中心对称
                old_x = (new_x + 0.5) * scale_x - 0.5
                old_y = (new_y + 0.5) * scale_y - 0.5

                # 向下取整，并转换数据类型
                old_x0 = int(np.floor(old_x))
                # 确保不出界
                old_x1 = min(old_x0 + 1, old_wide - 1)
                old_y0 = int(np.floor(old_y))
                old_y1 = min(old_y0 + 1 , old_hide - 1)

                # 计算双线性插值 算法
                temp0 = (old_x1 - old_x) * img[old_y0,old_x0,i] + (old_x - old_x0) * img[old_y0,old_x1,i]
                temp1 = (old_x1 - old_x) * img[old_y1,old_x0,i] + (old_x - old_x0) * img[old_y1,old_x1,i]
                new_img[new_y,new_x,i] = int((old_y1 - old_y) * temp0 + (old_y - old_y0) * temp1)
    return new_img
if True :
    img = cv2.imread("../第二周作业/image/img/lenna.png")
    dst = bilinear_interpolation(img,(900,900))
    cv2.imshow("builinear interp",dst)
    cv2.waitKey(0)