import numpy as np
import cv2

def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[0], out_dim[1]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    sh = float(src_h / dst_h)
    sw = float(src_w / dst_w)
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 原图和变换之后的图做一个中心对称
                src_y = float((dst_y + 0.5) * sh - 0.5)
                src_x = float((dst_x + 0.5) * sw - 0.5)
                # 防呆 如果目标图像换算到原图的坐标超出原图边界，就取原图边界的像素值
                # 临近点的坐标(x0, y0), (x1, y1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                # 代入双线性插值公式
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img

img = cv2.imread('city walk.jpg')
img_s = bilinear_interpolation(img, (600, 640))
cv2.imshow('bilinear_interpolation', img_s)
cv2.imshow('src', img)
cv2.waitKey()


