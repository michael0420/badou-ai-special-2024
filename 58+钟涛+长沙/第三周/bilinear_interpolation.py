import numpy as np
import cv2

#双线性插值

def b_i(img, out_dim):
    #获取图片长，宽以及通道
    src_h,src_w,src_c = img.shape
    #获取输出图片的长，宽
    dst_h = out_dim[0]
    dst_w = out_dim[1]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    #如果没有比例相同，直接返回原图片
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    #定义空数组
    out_img = np.zeros((dst_h, dst_w, src_c), dtype=np.uint8)
    #获取缩放比例
    zoom_x,zoom_y = float(src_h) / dst_h, float(src_w) / dst_w

    #迭代通道和图片像素
    for i in range(src_c):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # 获取当前点在原图片几何中心点
                x = (dst_x + 0.5)*zoom_x - 0.5
                y = (dst_y + 0.5) * zoom_y - 0.5

                #获取x1的坐标
                # np.floor()返回不大于输入参数的最大整数。（向下取整）
                x1 = int(np.floor(x))
                y1 = int(np.floor(y))

                #获取x2 坐标,原理就是上下相差为1，但是又不能超过原图最大值
                x2 = min(x1 + 1, src_w - 1)
                y2 = min(y1 + 1, src_h - 1)

                #计算中间结果
                temp0 = (x2 - x) * img[y1, x1, i] + (x - x1) * img[y1 ,x2, i]
                temp1 = (x2 - x) * img[y2 , x1 , i] + (x - x1) * img[y2, x2,i]
                out_img[dst_y, dst_x, i] = int((y2 -y) * temp0 + (y - y1) * temp1)
    return out_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    desc_img = b_i(img, (800,800))
    cv2.imshow("bilinear interp", desc_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








