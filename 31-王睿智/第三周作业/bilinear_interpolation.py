"""
作业：双线性插值
"""
import cv2
import numpy as np


def bilinear_interpolation(img, out_dim):
    """
    使用双线性插值法实现图片缩放
    :param img: 原始图片路径
    :param out_dim: 目标图片宽高
    :return:
    """
    # 获取原始图片高、宽、通道
    src_h, src_w, channel = img.shape
    # 目标图片高、宽
    dst_h, dst_w = out_dim[1], out_dim[0]
    # 若是原始图片与目标图片大小一致，则直接复制图片返回
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    # 定义目标图片的宽、高、通道、数据类型
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    # 计算宽、高缩放比
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    # 遍历所有通道
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 计算每一个像素点
                # 使用中心对齐方式获取原始(x,y)坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 获取整形数据
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                # 获取下一个点时需要考虑是否在边界
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)
                # 横轴从左到右顺序是 x0, x ,x1,纵轴从下到上是 y0, y, y1
                # temp0的计算对应公式里Q11、Q21
                # temp1的计算对应Q12、Q22
                # 剩下的就是原公式中：(y2-y1) * f(R1)+ (y-y1) * f(R2)
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dst = bilinear_interpolation(img, (300, 300))
    cv2.imshow("bilinear interp", dst)
    cv2.waitKey(0)