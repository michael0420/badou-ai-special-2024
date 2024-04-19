import numpy as np
import cv2
def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape             # 获取原图的信息
    dst_h,dst_w,=out_dim[1],out_dim[0]            # 输出的图像的尺寸
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:         # 判断输出的图像的尺寸是否与原图一致
        return img.copy()
    dst_img=np.zeros((dst_h,dst_w,3),dtype=np.uint8)        # 创建新的图像，3通道，8位
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h    #算出扩大的比列
    for i in range(channel):                                    #分别遍历三个通道
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x=(dst_x+0.5)*scale_x-0.5             # 为了确保图片信息的准确性，把二者的中心重合
                src_y=(dst_y+0.5)*scale_y-0.5
                src_x0 = int(np.floor(src_x))             # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1, src_w - 1)       # 为了确保取的点不会超出边界
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                     # 利用插值算法计算不同点、不同通道的像素值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))     # 输出图像的尺寸为700x700
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
