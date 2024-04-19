
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Util import cv_imread

img_path = "206-田海龙-北京/第03周/img/lenna.png"
img_path = "D:\Desktop\maomi.jpg"
def get_img(img_path,flag):
    img = cv_imread(img_path,flag)

    return img

def sobel_img(flag):
    """
    图像sobel处理，用于显示图像的边缘
    param:flag: 0-灰度图，1-彩色图
    """
    img = get_img(img_path,flag)
    print(img.shape)
    # 通道转换，否则plt显示图片RGB顺序不正确
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    '''
    Sobel函数求完导数后会有负值，还有会大于255的值。
    而原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。
    因此要使用16位有符号的数据类型，即cv2.CV_16S。
    ''' 
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    # 数据在没处理之前，cv2无法显示，plt可以显示
    # cv2.imshow("x-sobel",x)
    # cv2.imshow("y-sobel",y)

    plt.subplot(121)
    plt.imshow(x)
    plt.subplot(122)
    plt.imshow(y)
    plt.show()

    '''
    在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
    否则将无法显示图像，而只是一副灰色的窗口。
    dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
    其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
    '''

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    absX=cv2.cvtColor(absX,cv2.COLOR_BGR2RGB)
    absY=cv2.cvtColor(absY,cv2.COLOR_BGR2RGB)

    print(absX.shape)

    # 数据在处理之后，cv2、plt都可以显示
    cv2.imshow("x",absX)
    cv2.imshow("y",absY)
    
    plt.subplot(121)
    plt.imshow(absX)
    plt.subplot(122)
    plt.imshow(absY)
    plt.show()

    '''
    由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
    。其函数原型为：
    dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
    其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
    gamma是加到最后结果上的一个值。
    '''

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    cv2.imshow("Result", dst)

    dst=cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)

    plt.imshow(dst)
    plt.show()
    
# RGB多通道sobel
sobel_img(1)

# 灰度图sobel
sobel_img(0)

cv2.waitKey()
cv2.destroyAllWindows()
