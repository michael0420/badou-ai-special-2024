import  numpy as np
import cv2

'''
实现双线性插值
'''
def bilinear_interpolation(srcImg,tarImgDim):
    '''
    :param srcImg: 读取的原图
    :param tarImgDim: 目标图片的长宽
    :return: 原图或插值后的目标图片
    '''

    #拿到原图的长 宽 通道
    sourceHight = srcImg.shape[0]
    sourceWide  = srcImg.shape[1]
    sourceChannel = srcImg.shape[2]

    #拿到目标图片的长宽
    targetHight = tarImgDim[1]
    targetWide  = tarImgDim[0]

    print('原图的高 = ',sourceHight,'宽 = ',sourceWide)
    print('目标图的高 = ', targetHight, '宽 = ', targetWide)

    if sourceHight == targetHight and sourceWide == targetWide :
        #如果目标图片和原图长宽一致，直接返回原图
        return srcImg.copy()

    targetImg = np.zeros((targetHight,targetWide,3),dtype = np.uint8)  #创建空白的目标图片

    #计算目标图片和原图x轴和y轴的比例
    scaleX = float(sourceWide) / targetWide          #宽为x轴
    scaleY = float(sourceHight) / targetHight        #高为y轴

    #开始循环，从第一层通道开始遍历每一个像素点
    for i in range(sourceChannel):
        for targetY in range(targetHight):
            for targetX in range(targetWide):

                #找到目标图片像素点和原图像素点的坐标
                sourceX = (targetX + 0.5) * scaleX - 0.5
                sourceY = (targetY + 0.5) * scaleY - 0.5

                #找出用于计算插值的点的坐标
                sourceX0 = int(np.floor(sourceX))
                sourceX1 = min(sourceX0 + 1 , sourceWide -1)

                sourceY0 = int(np.floor(sourceY))
                sourceY1 = min(sourceY0 + 1, sourceHight - 1)

                #计算插值
                temp0 = (sourceX1 - sourceX) * srcImg[sourceY0,sourceX0,i] + (sourceX - sourceX0) * srcImg[sourceY0,sourceX1,i]
                temp1 = (sourceX1 - sourceX) * srcImg[sourceY1,sourceX0,i] + (sourceX - sourceX0) * srcImg[sourceY1,sourceX1,i]

                targetImg[targetY,targetX,i] = int((sourceY1 - sourceY) * temp0 + (sourceY - sourceY0) * temp1 )

    return targetImg

if __name__ == '__main__' :
    sourceImg = cv2.imread("/Users/mac/Desktop/tuanzi.jpg")
    targetImg = bilinear_interpolation(sourceImg,(800,1064))
    cv2.imshow('团子',targetImg)
    cv2.waitKey()