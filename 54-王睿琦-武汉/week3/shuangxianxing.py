import numpy as  np
import cv2

def shuangxianxing_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape #取出宽高和通道数
    dst_h, dst_w = out_dim[1], out_dim[0] #规定输出图像的尺寸
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    # 若尺寸相同，直接返回原图
    dst = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    # 创建图片
    scale_x, scale_y = float(src_w)/ dst_w, float(src_h)/ dst_h
    #计算比例
    for n in range(channel):  # 对channel循环
        for dst_y in range(dst_h):  # 对height循环
            for dst_x in range(dst_w):  # 对width循环
                # 目标在源上的坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 计算在源图上四个近邻点的位置
                src_x_0 = int(np.floor(src_x)) #np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_y_0 = int(np.floor(src_y))
                src_x_1 = min(src_x_0 + 1, src_w - 1)
                src_y_1 = min(src_y_0 + 1, src_h - 1)

                # 双线性插值
                value0 = (src_x_1 - src_x) * img[src_y_0, src_x_0, n] + (src_x - src_x_0) * img[src_y_0, src_x_1, n]
                value1 = (src_x_1 - src_x) * img[src_y_1, src_x_0, n] + (src_x - src_x_0) * img[src_y_1, src_x_1, n]
                dst[dst_y, dst_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    return dst


if __name__ == '__main__':
    image1 = cv2.imread('yangmi.jpg')  # 读图
    img = cv2.resize(image1, (360, 640))  # 图片缩放
    dst = shuangxianxing_interpolation(img , (540, 960))
    cv2.imshow("img",img)
    cv2.imshow("dst",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()