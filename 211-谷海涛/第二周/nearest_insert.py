import cv2
import numpy as np


def nearest_insert(img, new_size):
    ori_height, ori_width, channels = img.shape    #获取原图片的高度、宽度及通道数
    new_height, new_width = new_size
    new_img = np.zeros((new_height, new_width, channels), img.dtype)   #创建新图片大小的矩阵
    h_ratio = new_height/ori_height     #计算放大比例
    w_ratio = new_width/ori_width
    for i in range(new_height):
        for j in range(new_width):
            x = int(i/h_ratio + 0.5)   #使用round进行四舍五入时候需注意边界处理
            y = int(j/w_ratio + 0.5)
            new_img[i, j] = img[x, y]

    return new_img


img = cv2.imread('1.png')
size= [1000, 1000]
nearest_insert_img = nearest_insert(img, size)
cv2.imwrite('nearest_insert.png', nearest_insert_img)