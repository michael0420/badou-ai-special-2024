import cv2
import numpy as np

def DoubleBilinear(img, out_size):
    source_h, source_w, channel = img.shape
    #得到输出图像的长宽
    out_w, out_h = out_size[0], out_size[1]
    #如果要求的图像和原图像一样大，就没必要插值了，直接输出
    if source_h == out_h and source_w == out_w:
        return img.copy()

    #建立输出要求大小的图像
    out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    #得到比例，方便找原图中的虚拟坐标点
    scale_x, scale_y = float(source_w)/out_w, float(source_h)/out_h

    #三通道，每个元素点循环
    for i in range(channel):
        for out_y in range(out_h):
            for out_x in range(out_w):
                #找虚拟坐标点
                src_x = (out_x + 0.5) * scale_x - 0.5
                src_y = (out_y + 0.5) * scale_y - 0.5

                #找虚拟坐标点的周围4个点的x，y坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, source_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, source_h - 1)

                #双线插值
                #找R1，R2
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                #通过R1和R2找到最终虚拟点P的像素值
                out_img[out_y, out_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    #输出图像
    return out_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    result = DoubleBilinear(img, (700, 700))
    cv2.imshow('Source Image', img)
    cv2.imshow('DoubleBilinear', result)
    cv2.waitKey()