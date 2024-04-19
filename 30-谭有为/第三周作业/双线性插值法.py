import  numpy as np
import  cv2
import matplotlib.pyplot as  plt

img=cv2.imread("F:/PNG/lenna.png")
cv2.imshow('Original image',img)#显示原图
hegiht,width,channels=np.shape(img)
#print(hegiht,width,channels)
billnear_itpl_img=np.zeros((700,700,channels),np.uint8) #创建空白图片
h_proportion=float(hegiht/700)
w_proportion=float(width/700) #计算目标图片与原图的比例
for s in range(channels):
    for i in range (700):
        for j in range (700):
            x=(i+0.5)*h_proportion-0.5
            y=(j+0.5)*w_proportion-0.5  #计算原图上的映射点，+0.5目的是使中心重合，结果可能是虚拟点
            x0=int(np.floor(x))    #对虚拟点的坐标值取整，可对应上原图的坐标
            y0=int(np.floor(y))    #np.floor函数，返回不大于输入参数的最大整数（还是浮点型，因此需要int转换为整型）
            x1=min(x0+1,width-1)     #i0+1 表示 i0相邻的另一个点
            y1=min(y0+1,hegiht-1)    #min函数做边界值处理，若坐标点超出原图范围，取原图的最外点
            #上面得出了x0,x1两个X坐标，y0、y1两个Y坐标，可得4个点，由此4个点的坐标可求出P点坐标
            temp0=(x1-x)*img[x0,y0,s]+(x-x0)*img[x0,y1,s]
            temp1=(x1-x)*img[x1,y0,s]+(x-x0)*img[x1,y1,s]    # 求出R1、R2点
            billnear_itpl_img[i,j,s]=(x1-x)*temp1+(x-x0)*temp0  #根据R1 R2求出P点
cv2.imshow('billnear_itpl image',billnear_itpl_img)
cv2.waitKey()












