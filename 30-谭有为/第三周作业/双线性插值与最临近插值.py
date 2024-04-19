import cv2
import numpy as np
import matplotlib.pyplot as  plt

def near_inter(img,h_w):
    h,w,channels=np.shape(img)
    h_proportion=h/h_w[0]
    w_proportion=w/h_w[1]
    new_img=np.zeros((h_w[0],h_w[1],channels),dtype=np.uint8)
    for i in range(h_w[0]):
        for j in range(h_w[1]):
            x=int(i*h_proportion+0.5)
            y=int(j*w_proportion+0.5)
            new_img[i,j]=img[x,y]
    return new_img

def billnear_inter(img,h_w):
    h,w,channels=np.shape(img)
    h_proportion=h/h_w[0]
    w_proportion=w/h_w[1]
    new_img=np.zeros((h_w[0],h_w[1],channels),dtype=np.uint8)
    for s in range(channels):
        for i in range(h_w[0]):
            for j in  range(h_w[1]):
                x=(i+0.5)*h_proportion-0.5
                y=(j+0.5)*w_proportion-0.5  #计算原图上的映射点，+0.5目的是使中心重合，结果可能是虚拟点
                x0=int(np.floor(x))    #对虚拟点的坐标值取整，可对应上原图的坐标
                y0=int(np.floor(y))    #np.floor函数，返回不大于输入参数的最大整数（还是浮点型，因此需要int转换为整型）
                x1=min(x0+1,w-1)     #i0+1 表示 i0相邻的另一个点
                y1=min(y0+1,h-1)    #min函数做边界值处理，若坐标点超出原图范围，取原图的最外点
                #上面得出了x0,x1两个X坐标，y0、y1两个Y坐标，可得4个点，由此4个点的坐标可求出P点坐标
                temp0=(x1-x)*img[x0,y0,s]+(x-x0)*img[x0,y1,s]
                temp1=(x1-x)*img[x1,y0,s]+(x-x0)*img[x1,y1,s]    # 求出R1、R2点
                new_img[i,j,s]=(x1-x)*temp1+(x-x0)*temp0
    return new_img

'''
if __name__=='__main__':
    img=cv2.imread("F:/PNG/lenna.png")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(221)
    plt.imshow(img)
    img1=near_inter(img,(1000,1000))
    plt.subplot(222)
    plt.imshow(img1)
    img2=billnear_inter(img,(1000,1000))
    plt.subplot(223)
    plt.imshow(img2)
    plt.show()
'''
if __name__=='__main__':
    img=cv2.imread("F:/PNG/lenna.png")
    cv2.imshow('Original image',img)
    img1=near_inter(img,(256,256))
    img2=billnear_inter(img,(256,256))
    cv2.imshow('Nearest neighbor interpolation method',img1)
    cv2.imshow('billnear_itpl image',img2)
    cv2.waitKey()
