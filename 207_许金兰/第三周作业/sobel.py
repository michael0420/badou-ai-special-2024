"""
@author: 207-xujinlan
sobel边缘检测
"""

import cv2

def sobel_graph(img):
    """
    sobel边缘检测
    :param img: 灰度化的图片
    :return:
    """
    x=cv2.Sobel(img,cv2.CV_16S,1,0)   #x方向求导
    y=cv2.Sobel(img,cv2.CV_16S,0,1)    #y方向上求导
    convert_x=cv2.convertScaleAbs(x)    #转换成uint8形式
    convert_y=cv2.convertScaleAbs(y)
    new_img=cv2.addWeighted(convert_x,0.5,convert_y,0.5,0)
    cv2.imshow('sobel_x', convert_x)
    cv2.imshow('sobel_y', convert_y)
    cv2.imshow('result', new_img)
    cv2.waitKey(0)
if __name__=='__main__':
    img=cv2.imread('lenna.png',0)    #读入一幅灰度化的图片
    sobel_graph(img)
