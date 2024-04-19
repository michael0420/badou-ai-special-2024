import numpy as np
import cv2


def bilinear(img, out_h, out_w):
    # 取出原图的长、宽、通道数
    src_h, src_w, channel = img.shape
    # 如果原图长宽=新图长宽 直接复制原图返回
    if src_h == out_h and src_w == out_w:
        return img.copy()
    # 计算原图和新图的长宽比例
    scale_w = src_w / out_w
    scale_h = src_h / out_h
    # 按新的长宽新建空白图
    out_img = np.zeros((out_h, out_w, channel), dtype=np.uint8)
    for c in range(channel):
        for y in range(out_h):
            for x in range(out_w):
                # 按中心点重合的算法 算出在原图的位置
                src_x = (x + 0.5) * scale_w - 0.5
                src_y = (y + 0.5) * scale_h - 0.5

                # 求出双线性差值取值的相邻4个点
                src_x0 = int(src_x)
                src_x1 = min(src_x0 + 1, src_w - 1)  # 临界值超出范围处理
                src_y0 = int(src_y)
                src_y1 = min(src_y0 + 1, src_h - 1)  # 临界值超出范围处理

                # 根据双线性差值的计算公式计算像素点的像素值
                # 1.先计算X方向的两个插值
                temp_y0 = img[src_y0, src_x0, c] * (src_x1 - src_x) + img[src_y0, src_x1, c] * (src_x - src_x0)
                temp_y1 = img[src_y1, src_x0, c] * (src_x1 - src_x) + img[src_y1, src_x1, c] * (src_x - src_x0)
                # 2.根据计算出X方向的两个插值 按Y方向进行计算 得到最终的插值
                out_img[y, x, c] = int(temp_y0 * (src_y1 - src_y) + temp_y1 * (src_y - src_y0))
    return out_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    print("图片生成中。。。。")
    dst = bilinear(img, 300, 200)
    print("图片生成完成！！！")
    cv2.imshow('bilinear', dst)
    cv2.waitKey()
