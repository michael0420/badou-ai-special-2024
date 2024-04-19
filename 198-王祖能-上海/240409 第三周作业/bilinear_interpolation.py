#  双线性插值缩放图片,减弱锯齿状，图片更模糊
import cv2
# import matplotlib.pyplot as plt
import numpy as np


def bilinear_interpolation(old, out_dim):
    src_h, src_w, src_c = old.shape
    print(old[src_h - 1, src_w - 1])  # 取出最右下角像素值，与转换后对比
    dst_h, dst_w, dst_c = out_dim[0], out_dim[1], src_c
    print('源图:%s x %s x %s, 目标图:%s x %s x %s' % (src_h, src_w, src_c, dst_h, dst_w, dst_c))
    scale_x, scale_y = float(dst_w / src_w), float(dst_h / src_h)
    print('缩放比例：%s x %s' % (scale_x, scale_y))
    if src_h == dst_h and src_w == dst_w:  # 检查尺寸，一致直接复制
        return old.copy()
    new = np.zeros([dst_h, dst_w, dst_c], dtype=np.uint8)
    for c in range(dst_c):  # 减少计算量可以进行灰度化
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                src_x = (dst_x + 0.5) / scale_x - 0.5  # 几何中心对称，各通道像素平移缩放
                src_y = (dst_y + 0.5) / scale_y - 0.5
                # 寻找(src_x,src_y)所在原图区域的临近点,左下Q00，右下Q10，左上Q01，右上Q11
                src_x0 = int(src_x)
                src_y0 = int(src_y)
                '''防溢出'''
                src_x1 = min(src_x0 + 1, src_w - 1)  # src_x0最大可取511，此时如果+1超过源图最大尺寸，采用min限制
                src_y1 = min(src_y0 + 1, src_h - 1)
                # 求R1点(src_x,src_y0)插值，(src_x0,src_y0),(src_x1,src_y0)位置坐标与图像矩阵行列相反
                temp1 = (src_x1 - src_x) * old[src_y0, src_x0, c] + (src_x - src_x0) * old[src_y0, src_x1, c]
                # 求R2点(src_x,src_y1)插值，(src_x0,src_y1),(src_x1,src_y1)
                temp2 = (src_x1 - src_x) * old[src_y1, src_x0, c] + (src_x - src_x0) * old[src_y1, src_x1, c]
                # 求P点(src_x,src_y)插值，(src_x,src_y0),(src_x,src_y1)
                new[dst_y, dst_x, c] = int((src_y1 - src_y) * temp1 + (src_y - src_y0) * temp2)

    return new

# 使用自编的.py作为模块给别人import提供便利，有防止一导入模块直接运行整个脚本


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img1 = bilinear_interpolation(img, [400, 700])  # h * w 高 × 宽
    print(img1[399, 699])  # 转换后最右下角值与源图对比，周圈有黑点【0,0,0】
    cv2.imshow('bili_interp', img1)
    cv2.waitKey()
