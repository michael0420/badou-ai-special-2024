# 老师的代码
# import cv2
# import numpy as np
# def function(img):
#     height,width,channels =img.shape
#     emptyImage=np.zeros((800,800,channels),np.uint8)
#     sh=800/height
#     sw=800/width
#     for i in range(800):
#         for j in range(800):
#             x=int(i/sh + 0.5)  #int(),转为整型，使用向下取整。
#             y=int(j/sw + 0.5)
#             emptyImage[i,j]=img[x,y]
#     return emptyImage
#
# # cv2.resize(img, (800,800,c),near/bin)
#
# img=cv2.imread("lenna.png")
# zoom=function(img)
# print(zoom)
# print(zoom.shape)
# cv2.imshow("nearest interp",zoom)
# cv2.imshow("image",img)
# cv2.waitKey(0)
# 自己的代码
import time

import cv2
import cv2 as cv
import numpy as np


def nearest_interp(img, new_height, new_width):
    height, width, channels = img.shape  # 获取图像的高、宽、通道数
    emptyImage = np.zeros((new_height, new_width, channels), np.uint8)  # 创建一个新的空白图像
    sh = new_height / height  # 计算高度的缩放比例
    sw = new_width / width  # 计算宽度的缩放比例
    for i in range(new_height):  # 遍历新图像的每个像素
        for j in range(new_width):
            x = int(i / sh + 0.5)  # 计算原图像的坐标
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]  # 将原图像的像素赋值给新图像
    return emptyImage  # 返回新图像
def denoise_image():
    # 使用fastNlMeansDenoisingColored函数进行降噪
    dst = cv2.fastNlMeansDenoisingColored(newImg, None, 10, 10, 7, 21)
    return dst


img = cv.imread("lenna.png")  # 读取图像
new_height = int(input("请输入新图像的高度："))
new_width = int(input("请输入新图像的宽度："))
start = time.time()
newImg = nearest_interp(img,new_height,new_width)  # 对图像进行最近邻插值
end = time.time()
print(' took', end - start, 'seconds.')
denoised_img = denoise_image()  # 对新图像进行降噪处理
cv.imshow("nearest interp", newImg)  # 显示插值后的图像
cv.imshow("denoised image", denoised_img)  # 显示降噪后的图像
cv.imshow("image", img) # 显示原图像
cv.waitKey(0)  # 等待按键



