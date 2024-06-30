#!/usr/bin/env python
# coding: utf-8

# In[37]:


#读取图片并灰度化
import matplotlib.pyplot as plt
import math
import numpy as np

pic_path ='lenna.png'
img =plt.imread(pic_path)
# print(img)
img =img*255
img =img.mean(axis =-1)

# 高斯平滑
dx,dy =img.shape
sigma =1.50
n1 =1/(2*math.pi*sigma**2)
n2 =-1/(2*sigma**2)
dim =5
tmp =[i-dim//2 for i in range(dim)]
guassian_filter =np.zeros([dim,dim])
for i in range(dim):
    for j in range(dim):        
        guassian_filter[i,j] =n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
guassian_filter =guassian_filter/guassian_filter.sum()

# img_pad =np.zeros(img.shape[0])
tmp =dim//2
img_pad =np.pad(img,((tmp,tmp),(tmp,tmp)),'constant')
img_new =np.zeros(img.shape)
for i in range(dx):
    for j in range(dy):
        img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*guassian_filter)
        
plt.figure(1)
plt.imshow(img_new.astype(np.uint8),cmap='gray')
plt.axis('off')

#梯度计算
sobel_x =([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y =([[1,2,1],[0,0,0],[-1,-2,-1]])
img_tidu_x =np.zeros([dx,dy])
img_tidu_y =np.zeros(img.shape)
img_tidu =np.zeros(img.shape)
img_pad =np.pad(img_new,((1,1),(1,1)),'constant')


for i in range(dx):
    for j in range(dy):
        img_tidu_x[i,j] =np.sum(img_pad[i:i+3,j:j+3]*sobel_x)
        img_tidu_y[i,j] =np.sum(img_pad[i:i+3,j:j+3]*sobel_y)
        img_tidu[i,j] =np.sqrt(img_tidu_x[i,j]**2+img_tidu_y[i,j]**2)
plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8),cmap ='gray')
plt.axis('off')
                                          
#非极大值抑制
img_tidu[img_tidu_x==0]=0.0000001

angle =img_tidu_y/img_tidu_x
img_yizhi =np.zeros(img_tidu.shape)


for i in range(1,img_tidu.shape[0]-1):
    for j in range(1,img_tidu.shape[1]-1):
        tmp =img_tidu[i-1:i+3,j-1:j+3]
        flag =True
        if angle[i,j] >=1:
            n1 =(tmp[0,2]-tmp[0,1])/angle[i,j]+tmp[0,1]
            n2 =(tmp[2,0]-tmp[2,1])/angle[i,j]+tmp[2,1]
            if not (img_tidu[i,j] >n1 and img_tidu[i,j]>n2):
                flag =False
        if angle[i,j] <=-1:
            n1 =(tmp[0,1]-tmp[0,0])/angle[i,j]+tmp[0,1]
            n2 =(tmp[2,1]-tmp[2,2])/angle[i,j]+tmp[2,1]
            if not (img_tidu[i,j]>n1 and img_tidu[i,j]>n2):
                flag =False
        if angle[i,j] >0:
            n1 =(tmp[2,0]-tmp[1,0])/angle[i,j]+tmp[1,0]
            n2 =(tmp[0,2]-tmp[1,2])/angle[i,j]+tmp[1,2]
            if not (img_tidu[i,j]>n1 and img_tidu[i,j]>n2):
                flag =False
        if angle[i,j]<0:
            n1 =(tmp[1,0]-tmp[0,0])/angle[i,j]+tmp[1,0]
            n2 =(tmp[1,2]-tmp[2,2])/angle[i,j]+tmp[1,2]
            if not (img_tidu[i,j]>n1 and img_tidu[i,j]>n2):
                flag =False
        if flag:
            img_yizhi[i,j] =img_tidu[i,j]
plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
plt.axis('off')

# 双阈值检测
low_boundary =img_yizhi.mean()*0.5
high_boundary =low_boundary*3
zhan =[]
for i in range(1,img_yizhi.shape[0]-1):
    for j in range(1,img_yizhi.shape[1]-1):
        temp =img_yizhi[i-1:i+3,j-1:j+3]
        if img_yizhi[i,j]>high_boundary:
            img_yizhi[i,j]=255
            zhan.append([i,j])
        elif img_yizhi[i,j] <=low_boundary:
            img_yizhi[i,j] =0
while not len(zhan)==0:
    tx,ty =zhan.pop()
    a =img_yizhi[tx-1:tx+2,ty-1:ty+2]
    if (a[0,0]>low_boundary) and (a[0,0]<high_boundary):
        img_yizhi[tx-1,ty-1]=255
        zhan.append([tx-1,ty-1])
    if (a[0,1]>low_boundary) and (a[0,1]<high_boundary):
        img_yizhi[tx-1,ty] =255
        zhan.append([tx-1,ty])
    if (a[0,2]>low_boundary and a[0,2]<high_boundary):
        img_yizhi[tx-1,ty+1]=255
        zhan.append([tx-1,ty+1])
    if (a[1,0]>low_boundary) and (a[1,0]<high_boundary):
        img_yizhi[tx,ty-1]=255
        zhan.append([tx,ty-1])
    if (a[1,2]>low_boundary) and (a[1,2]<high_boundary):
        img_yizhi[tx,ty+1] =255
        zhan.append([tx,ty+1])
    if (a[2,0]>low_boundary) and (a[2,0]<high_boundary):
        img_yizhi[tx+1,ty-1] =255
        zhan.append([tx+1,ty-1])
    if (a[2,1]>low_boundary) and (a[2,1]<high_boundary):
        img_yizhi[tx+1,ty]=255
        zhan.append([tx+1,ty])
    if (a[2,2]>low_boundary) and (a[2,2]<high_boundary):
        img_yizhi[tx+1,ty+1]=255
        zhan.append([tx+1,ty+1])
        
for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i,j] !=0 and img_yizhi[i,j] !=255:
            img_yizhi[i,j] =0
plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
plt.axis('off')    

