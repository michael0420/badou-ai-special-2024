import cv2
import numpy as np
import  matplotlib.pyplot as plt
import  math





def CannyThreshold(lowthreshold):





    # Canny 边缘检测：优化程序
    detected_edges= cv2.Canny(
        img_gray,
        lowthreshold,
        lowthreshold*ratio,
        apertureSize=kernel_size




    )
    # 按位与操作，对于每个像素，将两幅输入图像相应位置的像素
    dst = cv2.bitwise_and(img,img,mask=detected_edges)
    cv2.imshow('canny res',dst)



lowthreshold = 0
max_lowthreshold = 100
ratio = 3
kernel_size=3
img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('canny res')


# 1.操作对象的名字， 2.对象所在面板名字 3.调节对象的默认值，也是调节对象 4.体调节范围 5. 调节所用的回调函数名
cv2.createTrackbar('min threshold','canny res',lowthreshold,max_lowthreshold,CannyThreshold)

CannyThreshold(0) #初始化

if cv2.waitKey(0) == 27:  # esc键对应的是27
    cv2.destroyAllWindows()




if __name__ == '__main__':

    #接口实现Canny算法
    # photo_path ='lenna.png'
    # img = cv2.imread(photo_path)
    # print("orgin",img)
    #
    # img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('canny',cv2.Canny(img_gray,200,300))  # 低阈值200 高阈值300
    # cv2.waitKey(0)

    CannyThreshold(0)


