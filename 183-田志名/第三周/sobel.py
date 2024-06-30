import cv2
import numpy as np
import math
#sobel算子
kernel1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

#卷积运算
def kernel(img,ker):
    h, w = img.shape
    result=np.zeros((h, w))        #此处不能写dtype=np.uint8,除非后续操作有abs，否则运算过程中没有负数了，向下溢出了，结果不对
    for i in range(1, h-1):
        for j in range(1, w-1):
            result[i-1,j-1] = min(255,abs(ker[0, 0] * img[i - 1, j - 1] + ker[0, 1] * img[i - 1, j] + ker[0, 2] * img[i - 1, j + 1]+\
                                          ker[1, 0] * img[i, j - 1] + ker[1, 1] * img[i, j] + ker[1, 2] * img[i, j + 1]+\
                                          ker[2, 0] * img[i + 1, j - 1] + ker[2, 1] * img[i + 1, j] + ker[2, 2] * img[i + 1, j + 1]))
    return np.uint8(result)

#将x轴与y轴的结果合并
def sobelxy(img1,img2):
    h,w=img1.shape
    gray_xy=np.zeros([h,w],dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            gray_xy[i,j]=math.sqrt(pow(img1[i,j],2)+pow(img2[i,j],2))
    return gray_xy


if __name__=="__main__":

    '''
    image=cv2.imread("lenna.png")
    red,green,blue=cv2.split(image)
    redx=kernel(red,kernel1)
    redy=kernel(red,kernel2)
    redxy=sobelxy(redx,redy)

    greenx=kernel(green,kernel1)
    greeny=kernel(red,kernel2)
    greenxy=sobelxy(greenx,greeny)

    bluex=kernel(blue,kernel1)
    bluey=kernel(blue,kernel2)
    bluexy=sobelxy(bluex,bluey)

    imgaexy=cv2.merge((redxy,greenxy,bluexy))
    cv2.imshow("qwe",imgaexy)
    cv2.imwrite("123456.png",imgaexy)
    '''

    img=cv2.imread("lenna.png")
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_x=kernel(gray_img, kernel1)
    gray_y=kernel(gray_img, kernel2)
    gray_xy=sobelxy(gray_x,gray_y)
    cv2.imshow("gray",gray_img)
    cv2.imshow("gray_x",gray_x)
    cv2.imshow("gray_y",gray_y)
    cv2.imshow("gray_xy",gray_xy)
    cv2.waitKey(0)

    #简洁版
    img = cv2.imread("lenna.png", 0)
    #原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。因此要使用16位有符号的数据类型，即cv2.CV_16S。
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)     #（1,0）求x方向的一阶导数
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)     #（0,1）求y方向的一阶导数
    #在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。否则将无法显示图像，而只是一副灰色的窗口
    absX = cv2.convertScaleAbs(x)            #cv2.convertScaleAbs（）用于线性转换和绝对值转换，对像素进行缩放和平移，此处只是将类型转成uint8
    absY = cv2.convertScaleAbs(y)
    #由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
    #cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0) #0.5是权重，最后的0是偏移，加到最后结果上的一个值
    cv2.imshow("absX", absX)
    cv2.imshow("absY", absY)
    cv2.imshow("Result", dst)
    cv2.waitKey(0)
    #修改地方：11行与18行，11行先不规定uint8,18行转成uint8,否则计算过程中存在溢出，使结果不准确
