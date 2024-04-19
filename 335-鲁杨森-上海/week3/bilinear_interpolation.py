"""
双线性插值 bilinear interpolaton
python implementation of bilinear interpolation
"""
import cv2
import numpy as np
def bilinear_interpolation(img, out_dim):
    src_h, src_w, channels = img.shape  # 取出原图尺寸
    dst_h, dst_w = out_dim[0], out_dim[1]  # 目标图尺寸
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, channels), dtype=np.uint8)  # uint8:8位无符号数(范围在[0,255])
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h  # 按比例对应
    for i in range(channels):  # 遍历三通道
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 几何中心对称（调整精度）
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 找原图坐标（找到将用于计算插值的点的坐标）
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                # 双线性插值公式
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


# if __name__ == '__main__':
#     img = cv2.imread("C:/Users/16040/Desktop/CV_/badou-ai-special-2024/335-鲁杨森-上海/week3/lenna.png")  # 读取图片
#     dst = bilinear_interpolation(img, (700, 700))  # 调用定义函数
#     cv2.imshow("bilinear interpolation", dst)  # 打印目标图
#     cv2.imshow("image", img)  # 打印原图
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
