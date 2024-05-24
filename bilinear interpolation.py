import numpy as np
import cv2
import time

def bilinear_interpolation_1(img, new_height, new_width):
    height, width, channels = img.shape # 获取图像的高、宽、通道数
    emptyImage = np.zeros((new_height, new_width, channels), np.uint8) # 创建一个新的空白图像
    sh = height / new_height # 计算高度的缩放比例
    sw = width / new_width
    for c in range(channels): # 遍历通道
        for i in range(new_height): # 遍历新图像的高
            for j in range(new_width): # 遍历新图像的宽
                x = (i + 0.5) * sh - 0.5 # 中心重合
                y = (j + 0.5) * sw - 0.5
                x1 = int(np.floor(x)) # 向下取整
                y1 = int(np.floor(y))
                x2 = min(x1 + 1, height - 1) # 防止越界
                y2 = min(y1 + 1, width - 1)
                emptyImage[i, j, c] = int((x2-x)*(y2-y)*img[x1, y1, c] + (x2-x)*(y-y1)*img[x1, y2, c] + (x-x1)*(y2-y)*img[x2, y1, c] + (x-x1)*(y-y1)*img[x2, y2, c])
    return emptyImage
def bilinear_interpolation_2(img, new_height, new_width):
    height, width, channels = img.shape
    dst_img = np.zeros((new_height, new_width, channels), np.uint8)
    scale_x = width / new_width
    scale_y = height / new_height

    for c in range(channels):
        for i in range(new_height):
            for j in range(new_width):
                src_x = (j + 0.5) * scale_x - 0.5
                src_y = (i + 0.5) * scale_y - 0.5
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0 + 1, width - 1)
                src_y1 = min(src_y0 + 1, height - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, c] + (src_x - src_x0) * img[src_y0, src_x1, c]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, c] + (src_x - src_x0) * img[src_y1, src_x1, c]
                dst_img[i, j, c] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")  # 读取图像
    new_height = int(input("请输入新图像的高度："))
    new_width = int(input("请输入新图像的宽度："))

    # 测试第一种实现方式
    start = time.time()
    newImg1 = bilinear_interpolation_1(img, new_height, new_width)
    end = time.time()
    print('Method 1 took', end - start, 'seconds.')

    # 测试第二种实现方式
    start = time.time()
    newImg2 = bilinear_interpolation_2(img, new_height, new_width)
    end = time.time()
    print('Method 2 took', end - start, 'seconds.')

    cv2.imshow('bilinear interp 1', newImg1)
    cv2.imshow('bilinear interp 2', newImg2)
    cv2.waitKey()