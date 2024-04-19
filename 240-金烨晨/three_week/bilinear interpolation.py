import numpy as np
import cv2

def bilinear_interpolation(img,image_size):
    src_h, src_w, src_c = img.shape     # 获取输入图像的高度、宽度和通道数
    dst_h, dst_w = image_size[0], image_size[1]     # 获取输出图像的目标高度和宽度
    if src_h == dst_h and src_w == dst_w:       # 如果输入图像和目标图像尺寸相同，则直接返回输入图像的副本
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, src_c), np.uint8)     # 创建一个全零数组，用于存储双线性插值后的图像
    scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h      # 计算宽度和高度的缩放比例
    for i in range(src_c):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1, src_w-1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1, src_h-1)

                temp0 = (src_x1-src_x) * img[src_y0,src_x0,i] + (src_x-src_x0) * img[src_y0,src_x1,i]   # 像素值
                temp1 = (src_x1-src_x) * img[src_y1,src_x0,i] + (src_x-src_x0) * img[src_y1,src_x1,i]   # 像素值
                dst_img[dst_y, dst_x, i] = int((src_y1-src_y) * temp0 + (src_y-src_y0) * temp1)

    return  dst_img


if __name__=='__main__':
    img = cv2.imread("../lenna.png")
    dst = bilinear_interpolation(img,(800,800))
    cv2.imshow("OD1",img)
    cv2.imshow("bilinear interpolation1", dst)

    img2 = cv2.imread("../roses-1566792_640.jpg")
    dst2 = bilinear_interpolation(img2,(800,800))
    cv2.imshow("OD2",img2)
    cv2.imshow("bilinear interpolation2", dst2)
    cv2.waitKey()
