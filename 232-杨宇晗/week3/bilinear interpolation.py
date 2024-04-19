"""
@author: Hanley-Yang

彩色图像的双线性插值
"""

import numpy as np
import cv2

def bilinear_interpolation(img, outDim):
    srcH, srcW, channel = img.shape
    dstH, dstW = outDim[0], outDim[1]
    #如果原图像和输出图像尺寸一致，则直接返回原图像
    if srcH == dstH and srcW == dstW:
        return img.copy()
    dstImg = np.zeros((dstH,dstW,3),dtype=np.uint8)
    scaleX,scaleY = float(srcW) / dstW, float(srcH) / dstH

    for i in range(channel):
        for dstY in range(dstH):
            for dstX in range(dstW):

                #找到输出图片的像素对应的原图像的像素坐标x和y
                #使原图像和输出图像的几何中心对应
                #srcX + 0.5 = (dstX + 0.5) * scale
                srcX = (dstX + 0.5) * scaleX - 0.5
                srcY = (dstY + 0.5) * scaleY - 0.5

                #根据原图像对应的像素坐标进行插值计算
                #X0,X1,Y0,Y1，坐标两两组合对应着最邻近的四个像素值
                srcX0 = int(np.floor(srcX)) #np.floor向下取整
                srcX1 = min(srcX0 + 1,srcW - 1) #srcW-1保证不越界
                srcY0 = int(np.floor(srcY))
                srcY1 = min(srcY0 + 1,srcH - 1)

                #计算插值，求对应原图像像素所在的x的直线,再算出直线x上的(x,y)像素对象
                temp0 = (srcX1 - srcX) * img[srcY0,srcX0,i] + (srcX-srcX0)*img[srcY0,srcX1,i]
                temp1 = (srcX1 - srcX) * img[srcY1,srcX0,i] + (srcX-srcX0)*img[srcY1,srcX1,i]
                dstImg[dstY,dstX,i] = int((srcY1-srcY) * temp0 + (srcY - srcY0) * temp1)

    return dstImg

if __name__ == '__main__':
    img = cv2.imread('Euphonium.png')
    dst = bilinear_interpolation(img,(1080,1920))

    # 保存处理后的图片到当前目录
    cv2.imwrite("bilinear interpolation.png",dst)
    cv2.imshow('bilinear interpolation',dst)

    cv2.waitKey()






