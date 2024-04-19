'''
将图像转换灰值化转化和二值化转化
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

class MyImg:
    # 源图像
    img = None
    # 灰质化转换后图像
    grayImg = None
    # 二值化转换图像
    binImg = None

    def __init__(self,path):
        self.img = cv2.imread(path)

    def showSrcImg(self):
        imgTmp = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        plt.subplot(2,2,1)
        plt.imshow(imgTmp)

    def toGrayAlgorithm(imgUnit):
        return 0.11 * imgUnit[0] + 0.59 * imgUnit[1] + 0.33 * imgUnit[2]

    def baseFunc(self,func,mode='gray'):
        h,w = self.img.shape[:2]
        dstImg = np.zeros([h,w],self.img.dtype)
        for i in range(h):
            for j in range(w):
                if mode == 'gray':
                    dstImg[i,j] = func(self.img[i,j])
                else:
                    dstImg[i,j] = func(self.grayImg[i,j])
        return dstImg

    def showGrayImg(self):
        plt.subplot(2,2,2)
        plt.imshow(self.grayImg,cmap='gray')

    def toBinaryAlgorithm(imgUnit):
        if imgUnit/255 > 0.5:
            return 1
        else:
            return 0

    def showBinaryImg(self):
        plt.subplot(2,2,3)
        plt.imshow(self.binImg,cmap='gray')

myImg = MyImg('./lenna.png')
myImg.showSrcImg()

myImg.grayImg = myImg.baseFunc(MyImg.toGrayAlgorithm)
myImg.showGrayImg()

myImg.binImg = myImg.baseFunc(MyImg.toBinaryAlgorithm,mode='binary')
myImg.showBinaryImg()

plt.show()














