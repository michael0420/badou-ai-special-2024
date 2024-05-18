import numpy as np
import cv2

# 实现双线性插值

def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape # 获得输入图像的高度，宽度与通道
    dst_h, dst_w = out_dim[1], out_dim[0] # out_dim[0]是输出图像的宽度 out_dim[1]是高度
    '''在图片规格表示为“300*400”时，通常第一个数字“300”代表宽度，第二个数字“400”代表高度。'''
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)

    if src_h == dst_h and src_w == dst_w:
        return  img.copy() # 目标大小和原图相同时 不需要放缩

    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8) # 初始化输出图像 默认值为0 3通道
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h # 计算放缩比例
    # 这里优先将src_w转为float 避免在除法时两个整数相除导致精度错误

    for i in range(channel): # 遍历3通道
        for dst_y in range(dst_h): # 遍历目标图像的y坐标
            for dst_x in range(dst_w): # 遍历目标图像的x坐标

                # 要求目标图像的点(dst_x, dst_y)
                # 为了保持几何中心对称 先使用公式计算出对应原图中的点(src_x, src_y)
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 再从点(src_x, src_y)计算出原图中对应做插值的四个点
                src_x0 = int(np.floor(src_x))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 做插值 现在x轴方向上做两次插值得到temp0,temp1
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                # 再在y轴方向上做一次插值得到最终结果
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return  dst_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img,(700,700))
    dst2 = cv2.resize(img, (700,700), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('DIY_bilinear interp',dst)
    cv2.imshow("bilinear interp", dst2)
    cv2.waitKey()