import numpy as np
import cv2


def bilinear(img, out_dim):
    # 获取输入图像的高度、宽度和通道数
    row, col, channels = img.shape
    # 根据out_dim获取输出图像的高度和宽度
    dst_row, dst_col = out_dim[1], out_dim[0]  # out_dim[1]为height也就是行数，   out_dim[0]为weight也就是列数
    # 打印输入输出图像的尺寸信息
    print("src_row, src_col = ", row, col)  # row是行数，原图和目标都是512   ，col是列数， 目标图为600
    print("dst_row, dst_col = ", dst_row, dst_col)

    # 如果输入图像和输出图像尺寸相同，则直接返回输入图像的副本
    if row == dst_row and col == dst_col:
        return img

    # 初始化输出图像，全零数组
    dst_img = np.zeros((dst_row, dst_col, 3), dtype=np.uint8)

    # 计算x和y方向的缩放比例
    scale_x, scale_y = float(row) / dst_row, float(col) / dst_col

    # 遍历输出图像的每个像素位置，进行双线性插值计算
    for i in range(channels):
        for dst_x in range(dst_row):  # 先循环row ， 并且把row的方向当作x方向
            for dst_y in range(dst_col):
                # 计算输入图像中对应当前输出像素位置的原始坐标
                # 采用几何中心对称的方法
                src_x = (dst_x + 0.5) * scale_x - 0.5  # scale_x为 ： row / dst_row  即把row的方向当作x
                src_y = (dst_y + 0.5) * scale_y - 0.5   # scale_y为 ： col / dst_col  即把col的方向当作y

                # 确定参与插值计算的四个像素点的坐标
                src_x0 = int(np.floor(src_x))      # 向下取整得到整数坐标，即（X0,Y0）是最接近（0，0）的点
                src_y0 = int(np.floor(src_y))

                src_x1 = min(src_x0 + 1, col - 1)  # 当x0+1大于边界的时候，则取边界值col-1
                src_y1 = min(src_y0 + 1, row - 1)

                # 计算插值权重并进行插值运算,注意：这里先X方向后Y方向
                temp0 = (src_x1 - src_x) * img[src_x0, src_y0, i] + (src_x - src_x0) * img[src_x1, src_y0, i]  # X方向插值1
                temp1 = (src_x1 - src_x) * img[src_x0, src_y1, i] + (src_x - src_x0) * img[src_x1, src_y1, i]  # X方向插值2
                dst_img[dst_x, dst_y, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)            # Y方向插值

    return dst_img


input_img = cv2.imread("../week2/lenna.png", 1)
dst = bilinear(input_img, [1000, 512])  # 这里的out_dim是 通常认知的  长 X 宽

cv2.imshow("bilinear_dst", dst)
cv2.waitKey(0)
