import cv2
import numpy as  np
import math

img=cv2.imread('F:/PNG/lenna.png',cv2.IMREAD_GRAYSCALE) #读取灰色图片
cv2.imshow('Original image',img)  #原图

'''
#直接调用cv2.canny函数实现canny边缘检测
guass_img=cv2.GaussianBlur(img,(5,5),1) #高斯滤波，5，5表示滤波器大小   1表示标准差
canny_edges_img=cv2.Canny(guass_img,50,200)  #边缘检测   100,200表示阈值
cv2.imshow('canny_edges_img',canny_edges_img)
cv2.waitKey()
'''
#具体实现
#高斯平滑/高斯滤波
sigma=0.5  #高斯核的标准差
dim=5
guass_filter=np.zeros([dim,dim])  #生成卷积核
temp=[]
for i in range(dim):
    temp0=i-dim//2
    temp.append(temp0)
print(temp)   #生成一个符合高斯分布的数组
parameter1=1/(2*math.pi*sigma**2)
parameter2=-1/(2*sigma**2)  #二维高斯分布公式中的参数
for i in  range(dim):
    for j in  range(dim):
        guass_filter[i,j]=parameter1*math.exp(parameter2*(temp[i]**2+temp[j]**2))#执行高斯平滑公式
#guass_filter = guass_filter / guass_filter.sum()  #归一化，可不做这一步
print(guass_filter)
h,w=img.shape
Gaussian_smoothing_img=np.zeros(img.shape)
temp=dim//2   #//表示取整  为了做全卷积，若卷积核是5*5  边缘填充上下左右都加2   若卷积核是3*3  边缘填充上下左右都加1
img_pad=np.pad(img,((temp,temp),(temp,temp)),'constant')  #np.pad函数  边缘填充  参考文档：https://www.cnblogs.com/shuaishuaidefeizhu/p/14179038.html
for i in range(h):
    for j in range (w):
        Gaussian_smoothing_img[i,j]=np.sum(img_pad[i:i+dim,j:j+dim]*guass_filter) #用边缘填充后的图片和卷积核做卷积
Gaussian_smoothing_img=Gaussian_smoothing_img.astype(np.uint8)  #做完卷积后的类型为浮点型，需要转换类型，图片才能显示
cv2.imshow('Gaussian_smoothing_img',Gaussian_smoothing_img)  #高斯平滑后的图


#利用sobel算子 求梯度tan值
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_x=np.zeros(img.shape)
img_y=np.zeros(img.shape)
img_gradient=np.zeros(img.shape)
img_pad=np.pad(img,((1,1),(1,1)),'constant')  #边缘填充，因为卷积核为3*3  因此上下左右各填充一行即可做全卷积
for i in  range(h):
    for j in range(w):
        img_x[i,j]=np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_x)
        img_y[i,j]=np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_y)   #分别对X轴和Y轴的边缘进行检测
        img_gradient[i,j]=np.sqrt(img_x[i,j]**2+img_y[i,j]**2)
img_x[img_x==0] =0.00000001  #下面代码img_x要做除数 因此不能为0
tan=img_y/img_x   #tan=y/x
print(tan)
#img_gradient=img_gradient.astype(np.uint8)
cv2.imshow('img_gradient',img_gradient.astype(np.uint8))   #sobel边缘检测后的图


#非极大值抑制
img_inhibition=np.zeros(img_gradient.shape)
print(temp)
for i in range(1,h-1):  #图片边缘不处理
    for j in range(1,w-1):
        flag=True
        temp=img_gradient[i-1:i+2,j-1:j+2]  #8邻域
        if tan[i,j]<=-1:
            num1=(temp[0,1]-temp[0,0])/tan[i,j]+temp[0,1]    #已知tan=y/x  因此 y/tan=x
            num2=(temp[2,1]-temp[2,2])/tan[i,j]+temp[2,1]
            if not(img_gradient[i,j]>num1 and img_gradient[i,j]>num2):
                flag=False
        elif tan[i,j]>=1:
            num1=(temp[0,2]-temp[0,1])/tan[i,j]+temp[0,1]
            num2=(temp[2,0]-temp[2,1])/tan[i,j]+temp[2,1]
            if not(img_gradient[i,j]>num1 and img_gradient[i,j]>num2):
                flag=False
        elif tan[i,j]>0:
            num1=(temp[0,2]-temp[1,2])*tan[i,j] + temp[1,2]
            num2=(temp[2,0]-temp[1,0])*tan[i,j] + temp[1,0]
            if not(img_gradient[i,j] > num1 and img_gradient[i,j] > num2):
                flag=False
        elif tan[i,j] < 0:
            num1=(temp[1,0]-temp[0,0])*tan[i,j] + temp[1, 0]
            num2=(temp[1,2]-temp[2,2])*tan[i,j] + temp[1, 2]
            if not(img_gradient[i,j] > num1 and img_gradient[i,j] > num2):
                flag=False
        if flag==True:
            img_inhibition[i,j]=img_gradient[i,j]
img_inhibition=img_inhibition.astype(np.uint8)
cv2.imshow('img_inhibition',img_inhibition)  #非极大值抑制后的图

#双阈值检测
Low_threshold=100
High_threshold=200  #设置双阈值
zhan=[]    #用于存储强边缘点
h,w=img_inhibition.shape
print(h,w)
for i in range(1,h-1):
    for j in range(1,w-1):
        if img_inhibition[i,j]>=High_threshold:   #梯度值大于高阈值，标记为强边缘
            img_inhibition[i,j]=255
            zhan.append([i,j])
        elif img_inhibition[i,j]<=Low_threshold:   #梯度值小于低阈值，标记为不是边缘
            img_inhibition[i,j]=0
print('zhan',zhan)
##抑制孤立低阈值点
while not len(zhan)==0:
    tmp1,tmp2=zhan.pop() #取出强边缘
    a=img_inhibition[tmp1-1:tmp1+2,tmp2-1:tmp2+2]  #强边缘的8邻域
    if (a[0,0]<High_threshold) and (a[0,0]>Low_threshold):
        img_inhibition[tmp1-1,tmp2-1]=255  # 如果弱边缘在强边缘的8邻域内，把弱边缘变为强边缘
        zhan.append([tmp1-1,tmp2-1])
    if (a[0,1]<High_threshold) and (a[0,1]>Low_threshold):
        img_inhibition[tmp1-1, tmp2]=255
        zhan.append([tmp1-1, tmp2])
    if (a[0,2]<High_threshold) and (a[0,2]>Low_threshold):
        img_inhibition[tmp1-1, tmp2+1]=255
        zhan.append([tmp1-1,tmp2+1])
    if (a[1,0]<High_threshold) and (a[1,0]>Low_threshold):
        img_inhibition[tmp1,tmp2-1]=255
        zhan.append([tmp1, tmp2-1])
    if (a[1,2]<High_threshold) and (a[1,2]>Low_threshold):
        img_inhibition[tmp1,tmp2+1] = 255
        zhan.append([tmp1, tmp2+1])
    if (a[2,0]<High_threshold) and (a[2,0]>Low_threshold):
        img_inhibition[tmp1+1,tmp2-1]=255
        zhan.append([tmp1+1,tmp2-1])
    if (a[2,1]<High_threshold) and (a[2,1]>Low_threshold):
        img_inhibition[tmp1+1,tmp2]=255
        zhan.append([tmp1+1,tmp2])
    if (a[2,2]<High_threshold) and (a[2,2]>Low_threshold):
        img_inhibition[tmp1+1,tmp2+1]=255
        zhan.append([tmp1+1,tmp2+1])

#处理噪声
for i in  range(h):
    for j in range(w):
        if img_inhibition[i,j]!=0 and img_gradient[i,j]!=255:
            img_gradient[i,j]=0


img_inhibition_after=img_inhibition.astype(np.uint8)
cv2.imshow('img_inhibition_after',img_inhibition_after)   #双阈值检测后的图
cv2.waitKey(0)
cv2.destroyAllWindows()





