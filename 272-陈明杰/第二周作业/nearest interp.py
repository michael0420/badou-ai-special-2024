# 导入对应的模块
import cv2
import numpy as np
import sys


def nearestInterp(img: np.ndarray, newSize: int):
    # 把图片的高度，宽度，通道提取出来
    high, width, channels = img.shape
    # 创建一个目标大小的空矩阵
    emptyImg = np.zeros((newSize, newSize, channels), np.uint8)

    # 计算新图像与旧图像的高度和宽度的放大/缩小倍数
    sh = newSize / high
    sw = newSize / width

    # 利用最邻近插值法填写新图像矩阵的所有位置
    for i in range(0, newSize):
        for j in range(0, newSize):
            # 计算新矩阵中[i,j]位置对应到原矩阵中的最邻近的下标
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            # emptyImg[i,j]=img[x,y]
            # 用原矩阵中最邻近的位置的像素填写新矩阵
            emptyImg[i][j] = img[x][y]
    return emptyImg


def main():
    img = cv2.imread("lenna.png")
    # print(type(img))
    # 输入的尺寸注意不能越界
    newSize = int(input("请输入放大后的尺寸："))
    zoom = nearestInterp(img, newSize)
    # 显示原矩阵
    cv2.imshow("before", img)
    # 显示插值后的矩阵
    cv2.imshow("after", zoom)
    cv2.waitKey(0)


main()



# def nearestInterp(img: np.ndarray):
#     h,w,c=img.shape
#     sh=800/h
#     sw=800/w
#     emptyImg=np.zeros((800,800,c),np.uint8)
#     for i in range(800):
#         for j in range(800):
#             x=int(i/sh+0.5)
#             y=int(j/sw+0.5)
#             emptyImg[i,j]=img[x,y]
#     return emptyImg
#
# def main():
#     img=cv2.imread("lenna.png")
#     zoom=nearestInterp(img)
#     cv2.imshow("before",img)
#     cv2.imshow("after",zoom)
#     cv2.waitKey(0)
#
# main()
