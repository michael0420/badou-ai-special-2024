# 练习双线性插值
import numpy as np
import cv2

def bilinear_interpolation(img, in_img):
    scr_height, scr_width, channels = img.shape  # 获取原图像的高、宽和通道
    dst_height, dst_width = in_img[1], in_img[0]
    if dst_height == scr_height and dst_width == scr_width:  # 判断目标图像高宽是否和原图像高宽相同
        return img.copy()
    dst_img = np.zeros((dst_height, dst_width, channels), img.dtype)
    scale_y = float(scr_height) / dst_height  # 获取原图与目标图高度上的比例
    scale_x = float(scr_width) / dst_width  # 获取原图与目标图宽度上的比例

    for i in range(channels):
        for j in range(dst_height):
            for k in range(dst_width):
                scr_y = (j + 0.5) * scale_y - 0.5  # 几何中心重合
                scr_x = (k + 0.5) * scale_x - 0.5

                scr_x0 = int(np.floor(scr_x))  # np.floor(scr_x)表示返回不大于scr_x的最大整数(向下取整)
                scr_x1 = min(scr_x0 + 1, scr_width - 1) # 防止超过索引边界
                scr_y0 = int(np.floor(scr_y))
                scr_y1 = min(scr_y0 + 1, scr_height - 1)

                # 套用双线性插值公式
                temp0 = (scr_x1 - scr_x) * img[scr_y0, scr_x0, i] + (scr_x - scr_x0) * img[scr_y0, scr_x1, i]
                temp1 = (scr_x1 - scr_x) * img[scr_y1, scr_x0, i] + (scr_x - scr_x0) * img[scr_y1, scr_x1, i]
                dst_img[j, k, i] = int((scr_y1 - scr_y) * temp0 + (scr_y - scr_y0) * temp1)  # 色值取值为0 ~ 255 的整数取值，所以要取整处理
    return dst_img

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dst_img = bilinear_interpolation(img, (700, 700))
    cv2.imshow("dst_img", dst_img)
    cv2.waitKey(0)