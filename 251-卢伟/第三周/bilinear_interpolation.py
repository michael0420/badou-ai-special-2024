import cv2
import numpy as np


# 双线性插值
def bilinear_interpolation(new_h, new_w):
    # 读图片
    img = cv2.imread('../Lenna.jpg')

    h, w, c = img.shape
    scale_h = float(h) / new_h
    scale_w = float(w) / new_w
    dst_img = np.zeros((new_h, new_w, c), dtype=np.uint8)

    print("scale_x, scale_y = ", scale_w, scale_h)

    for i in range(c):
        for y in range(new_h):
            for x in range(new_w):
                src_x = (x + 0.5) * scale_w - 0.5
                src_y = (y + 0.5) * scale_h - 0.5

                # 确定四个点的x,y值
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0 + 1, w - 1)
                src_y1 = min(src_y0 + 1, h - 1)

                # 带入公式计算插值点的坐标
                fr0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                fr1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                # print(fr1,fr2)
                dst_img[y, x, i] = int((src_y1 - src_y) * fr0 + (src_y - src_y0) * fr1)

    cv2.imshow("dst_image", dst_img)
    cv2.waitKey()


bilinear_interpolation(500, 500)
