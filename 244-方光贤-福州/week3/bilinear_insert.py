import cv2
import numpy as np

def bilinear_insert(img, dim) :
    #设置目标图像和原图的图像属性
    src_h, src_w, channel = img.shape
    dst_h, dst_w = dim[0], dim[1]
    #打印图像属性
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    #复制一个新图像
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    #给新图像幅值相关属性
    dst_img = np.zeros((dst_h, dst_w,3), dtype=np.uint8)
    #建立原图和目标图的比例关系 设置为浮点型 保证计算精度
    scale_x = float(dst_w / src_w)
    scale_y = float(dst_h / src_h)

    #由于是彩图所以 通道为3 遍历三次 若为灰度图 通道为1 遍历一次即可
    for i in range(channel):
        for dst_x in range(dst_h):
            for dst_y in range(dst_w):
                # 做中心对齐 推导得到左右相加0.5
                src_x = (dst_x + 0.5) / scale_x - 0.5
                src_y = (dst_y + 0.5) / scale_y - 0.5
                #向下取整
                src_x0 = int(src_x)
                src_y0 = int(src_y)
                #防止越界 越界了可以取最大尺寸来计算
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)
                #计算双线性插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_insert(img, (700, 700))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
