import numpy as np
import cv2

'''
用python代码实现双线性插值的练习
'''


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape  # 提取高度 ( src_h)、宽度 ( src_w) 和颜色通道数 (channel)
    dst_h, dst_w = out_dim[0], out_dim[1]  #赋值dst_h, dst_w等于目标图像像素值
    print("sre_c,sre_c=", src_h, src_w)
    print("dst_h,dst_w=", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:   #如果原图像与目标图像X、Y相等，使用copy函数，复制
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)#创建空目标图像dst_img,指定数组的形状。dst_h代表目标图像的高度，dst_w代表宽度，3代表颜色通道数（假设RGB颜色空间）。dtype=np.uint8：指定数组元素的数据类型。
    # np.uint8代表 8 位无符号整数，是表示图像中像素值的常见数据类型。每个像素值的范围为 0 到 255。
    scale_x, scale_y = float(src_h) / dst_h, float(src_w) / dst_w  #通过将源图像尺寸除以目标图像尺寸来计算缩放因子scale_x，scale_y，该float()函数用于确保除法结果为浮点值。
    for i in range(channel):   #循环通道数
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5  #srcX+0.5=(dstX+0.5)*(srcX/dstX)
                src_y = (dst_y + 0.5) * scale_y - 0.5  #srcY+0.5=(dstY+0.5)*(srcY/dstY)


                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                '''
                计算源图像中当前位置 ( , ) 周围的四个相邻像素坐标 (src_x0,src_x1,src_y0,src_y1) 。该函数用于对浮点值进行向下舍入并将其转换为整数。该函数用于确保坐标不超出源图像的边界。
                '''
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
                '''
                对每个颜色通道执行双线性插值，i以计算当前目标像素的强度值。双线性插值是通过考虑源图像中的四个相邻像素 ( src_x0、src_x1、src_y0、src_y1) 来完成的。强度值计算为这些相邻像素的强度的加权平均值，其中权重由当前位置 ( src_x, src_y) 与相邻像素位置之间的距离确定
                '''

    return dst_img


img = cv2.imread("lenna.png")
dst = bilinear_interpolation(img, (700, 700))#目标图像
cv2.imshow('Original image', img)
cv2.imshow('bilinear interp', dst)
cv2.waitKey()
