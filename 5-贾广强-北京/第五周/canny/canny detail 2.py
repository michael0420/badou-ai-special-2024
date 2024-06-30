#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

if __name__ == '__main__':
    p_path = 'lenna.jpg'
    img = plt.imread(p_path)
    if p_path[-4:] == '.jpg':
        img = img * 255
    img = img.mean(axis=-1)
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(img)
    
    #高斯滤波
    sigma = 0.5
    dim = 5
    Gaussian_f = np.zeros([dim,dim])
    temp = [i-dim//2 for i in range(dim)]
    n1 = 1/(2*math.pi*sigma**2)
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_f[i,j] = n1*math.exp(n2*(temp[i]**2+temp[j]**2))
    Gaussian_f = Gaussian_f / Gaussian_f.sum()
    dx,dy = img.shape
    img_new = np.zeros((dx,dy))
    temp_0 = dim//2 
    img_pad = np.pad(img,((temp_0,temp_0),(temp_0,temp_0)),'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*Gaussian_f)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8),cmap='gray')
    
    #使用soble
    soble_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    soble_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    img_x = np.zeros(img_new.shape)
    img_y = np.zeros(img_new.shape)
    img_td = np.zeros(img_new.shape)
    img_pad = np.pad(img_new,((1,1),(1,1)),'constant')
    for i in range(dx):
        for j in range(dy):
            img_x[i,j] = np.sum(img_pad[i:i+3,j:j+3]*soble_x)
            img_y[i,j] = np.sum(img_pad[i:i+3,j:j+3]*soble_y)
            img_td[i,j] = np.sqrt(img_x[i,j]**2+img_y[i,j]**2)
    img_x[img_x == 0]=1e-8
    tan_1 = img_y / img_x
    plt.figure(2)
    plt.imshow(img_td.astype(np.uint8),cmap='gray')
    
    #非极大值抑制
    img_yz = np.zeros(img_td.shape)
    for i in range(1,dx-1):
        for j in range(1,dy-1):
            flag = True
            temp_ly = img_td[i-1:i+2,j-1:j+2]
            if tan_1[i,j] <=-1:
                num_1 = (temp_ly[0,1]-temp_ly[0,0])/tan_1[i,j] +temp_ly[0,1]
                num_2 = (temp_ly[2,1]-temp_ly[2,2])/tan_1[i,j] +temp_ly[2,1]
                if not(img_td[i,j] > num_1 and img_td[i,j] > num_2):
                    flag = False
            elif tan_1[i,j] >=1:
                num_1 = (temp_ly[0,2]-temp_ly[0,1])/tan_1[i,j] +temp_ly[0,1]
                num_2 = (temp_ly[2,0]-temp_ly[2,1])/tan_1[i,j] +temp_ly[2,1]
                if not(img_td[i,j] > num_1 and img_td[i,j] > num_2):
                    flag = False
            elif tan_1[i,j] >0:
                num_1 = (temp_ly[0,2]-temp_ly[1,2])*tan_1[i,j] +temp_ly[1,2]
                num_2 = (temp_ly[2,0]-temp_ly[1,0])*tan_1[i,j] +temp_ly[1,0]
                if not(img_td[i,j] > num_1 and img_td[i,j] > num_2):
                    flag = False
            elif tan_1[i,j] <0:
                num_1 = (temp_ly[1,0]-temp_ly[0,0])*tan_1[i,j] +temp_ly[1,0]
                num_2 = (temp_ly[1,2]-temp_ly[2,2])*tan_1[i,j] +temp_ly[1,2]
                if not(img_td[i,j] > num_1 and img_td[i,j] > num_2):
                    flag = False
            if flag:
                img_yz[i,j] = img_td[i,j]
    plt.figure(3)
    plt.imshow(img_yz.astype(np.uint8),cmap='gray')
    
    #双阀值检测
    lower_b = img_td.mean() *0.5
    high_b = lower_b *3
    zhan = []
    for i in range(1,img_yz.shape[0]-1):
        for j in range(1,img_yz.shape[1]-1):
            if img_yz[i,j] >= high_b:
                img_yz[i,j] = 255
                zhan.append([i,j])
            elif img_yz[i,j] <lower_b:
                img_yz[i,j] = 0
                
    while not len(zhan) == 0:
        temp_1,temp_2 = zhan.pop()
        a = img_yz[temp_1-1:temp_1+2,temp_2-1:temp_2+2]
        if(a[0,0]<high_b) and (a[0,0]>lower_b):
            img_yz[temp_1-1,temp_2-1] = 255
            zhan.append([temp_1-1,temp_2-1])
        if(a[0,1]<high_b) and (a[0,1]>lower_b):
            img_yz[temp_1-1,temp_2] = 255
            zhan.append([temp_1-1,temp_2])
        if(a[0,2]<high_b) and (a[0,2]>lower_b):
            img_yz[temp_1-1,temp_2+1] = 255
            zhan.append([temp_1-1,temp_2+1])
        if(a[1,0]<high_b) and (a[1,0]>lower_b):
            img_yz[temp_1,temp_2-1] = 255
            zhan.append([temp_1,temp_2-1] )  
        if(a[1,2]<high_b) and (a[1,2]>lower_b):
            img_yz[temp_1,temp_2+1] = 255
            zhan.append([temp_1,temp_2+1])   
        if(a[2,0]<high_b) and (a[2,0]>lower_b):
            img_yz[temp_1+1,temp_2-1] = 255
            zhan.append([temp_1+1,temp_2-1])
        if(a[2,2]<high_b) and (a[2,2]>lower_b):
            img_yz[temp_1+1,temp_2+1] = 255
            zhan.append([temp_1+1,temp_2+1])
    for i in range(img_yz.shape[0]):
        for j in range(img_yz.shape[1]):
            if img_yz[i,j] != 0 and img_yz[i,j]!=255:
                img_yz[i,j] = 0
    plt.figure(4)
    plt.imshow(img_yz.astype(np.uint8),cmap='gray')

