# 双线性插值
import numpy as np
import cv2


def two_line_insert(img, dist):
    # 获取图片宽、高、通道数
    src_h, src_w, channel = img.shape
    # 获取想要生成图片的宽、高
    dst_h, dst_w = dist[1], dist[0]
    print(src_h, src_w, channel)
    print(dst_h, dst_w)
    # 如果要生成的图片的宽、高与原图的宽高相等，则复制原图
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    # 建立空白目标图片
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    # 计算原图和目标图片的比例
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # (x+c) = (x1+c)*缩放比例
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                # 防止超出原图尺寸
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


if __name__ == '__main__':
    img = cv2.imread('../0327/lenna.png')  # 读取图片
    dst = two_line_insert(img, (700, 700))
    cv2.imshow('result', dst)
    cv2.waitKey()
