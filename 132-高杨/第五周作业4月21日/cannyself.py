import matplotlib.pyplot as plt
import math
import cv2
import numpy as np

def canny1(src_path,dim):

    img_orgin = plt.imread(src_path)

    if src_path[-4:]=='.png':
        img_orgin=img_orgin*255
    #利用均值求灰度图和 求出高斯核，并对原图做高斯滤波
    img_orgin=img_orgin.mean(axis=-1)
    sigma =0.5
    n1,n2=1/(2*math.pi*sigma**2),-1/(2*sigma**2)
    valuelist=[i-dim//2 for i in range(dim)]
    gaussian_kernel=np.zeros([dim,dim])
    for i in range(dim):
        for j in range(dim):
            gaussian_kernel[i,j]=n1*math.exp(n2*(valuelist[i]**2+valuelist[j]**2))
    gaussian_kernel=gaussian_kernel / gaussian_kernel.sum()
    img_new=np.zeros(img_orgin.shape)
    tmp=dim//2
    img_pad=np.pad(img_orgin,((tmp,tmp),(tmp,tmp)),'constant')
    for i in range(img_orgin.shape[0]):
        for j in range(img_orgin.shape[1]):
            img_new[i,j]=np.sum(img_pad[i:i+dim,j:j+dim]*gaussian_kernel)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8),cmap='gray')
    plt.axis('off')

    # 第二步 求梯度 ，
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_grad_x,img_grad_y,img_grad=np.zeros(img_orgin.shape),np.zeros(img_orgin.shape),np.zeros(img_orgin.shape)
    img_pad=np.pad(img_new,((1,1),(1,1)),'constant')  # 3//2 为1
    for i in range(img_orgin.shape[0]):
        for j in range(img_orgin.shape[1]):
            img_grad_x[i, j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_x)
            img_grad_y[i, j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_y)
            img_grad[i, j] = np.sqrt(img_grad_x[i, j]**2+img_grad_y[i, j]**2)
    img_grad_x[img_grad_x == 0]=0.00000001
    tansita=img_grad_y/img_grad_x
    plt.figure(2)
    plt.imshow(img_grad.astype(np.uint8),cmap='gray')
    plt.axis('off')

    dx,dy=img_orgin.shape
    # 第三步 非极大值抑制
    img_restrain=np.zeros(img_grad.shape)
    for i in range(1,dx-1):
        for j in range(1,dy-1):
            flag=True
            tmepmatrix=img_grad[i-1:i+2,j-1:j+2]
            if tansita[i,j]<=-1:
                nums1=(tmepmatrix[0,1]-tmepmatrix[0,0]) / tansita[i,j] +tmepmatrix[0,1]
                nums2=(tmepmatrix[2,1]-tmepmatrix[2,2]) / tansita[i,j] + tmepmatrix[2,1]
                if not (img_grad[i,j]>nums1 and img_grad[i,j]>nums2):
                    flag=False
            elif tansita[i, j] >= 1:
                nums1 = (tmepmatrix[0, 2] - tmepmatrix[0, 1]) / tansita[i, j] + tmepmatrix[0, 1]
                nums2 = (tmepmatrix[2, 0] - tmepmatrix[2, 1]) / tansita[i, j] + tmepmatrix[2, 1]
                if not (img_grad[i, j] > nums1 and img_grad[i, j] > nums2):
                    flag = False
            elif tansita[i, j] < 0:
                nums1 = (tmepmatrix[1, 0] - tmepmatrix[0, 0]) * tansita[i, j] + tmepmatrix[1, 0]
                nums2 = (tmepmatrix[1, 2] - tmepmatrix[2, 2]) * tansita[i, j] + tmepmatrix[1, 2]
                if not (img_grad[i, j] > nums1 and img_grad[i, j] > nums2):
                    flag = False
            elif tansita[i, j] > 0:
                nums1 = (tmepmatrix[0, 2] - tmepmatrix[1, 2]) * tansita[i, j] + tmepmatrix[1, 2]
                nums2 = (tmepmatrix[2, 0] - tmepmatrix[1, 0]) * tansita[i, j] + tmepmatrix[1, 0]
                if not (img_grad[i, j] > nums1 and img_grad[i, j] > nums2):
                    flag = False
            if flag:
                img_restrain[i,j]=img_grad[i,j]
    plt.figure(3)
    plt.imshow(img_restrain.astype(np.uint8), cmap='gray')
    plt.axis('off')



    # 第四步 双阈值
    low_boundary = img_grad.mean()*0.5  #确定阈值范围
    high_boundary =low_boundary*3
    stack=[]
    for i in range(1,img_restrain.shape[0]-1):
        for j in range(1, img_restrain.shape[1] - 1):
            if img_restrain[i,j]>=high_boundary:
                img_restrain[i,j]=255
                stack.append([i,j])
            elif img_restrain[i,j]<=low_boundary:
                img_restrain[i,j]=0

    while not len(stack)==0:
        temp_1,temp_2=stack.pop()
        a= img_restrain[temp_1-1:temp_1+2 , temp_2-1:temp_2+2]
        if (a[0,0]<high_boundary) and (a[0,0] >low_boundary ):
            img_restrain[temp_1-1,temp_2-1]=255
            stack.append([temp_1-1,temp_2-1])
        if (a[0,1]<high_boundary) and (a[0,1] >low_boundary ):
            img_restrain[temp_1-1,temp_2]=255
            stack.append([temp_1-1,temp_2])
        if (a[0,2]<high_boundary) and (a[0,2] >low_boundary ):
            img_restrain[temp_1-1,temp_2+1]=255
            stack.append([temp_1-1,temp_2+1])
        if (a[1,0]<high_boundary) and (a[1,0] >low_boundary ):
            img_restrain[temp_1,temp_2-1]=255
            stack.append([temp_1,temp_2-1])
        if (a[1,2]<high_boundary) and (a[1,2] >low_boundary ):
            img_restrain[temp_1,temp_2+1]=255
            stack.append([temp_1,temp_2+1])
        if (a[2,0]<high_boundary) and (a[2,0] >low_boundary ):
            img_restrain[temp_1+1,temp_2-1]=255
            stack.append([temp_1+1,temp_2-1])
        if (a[2,1]<high_boundary) and (a[2,1] >low_boundary ):
            img_restrain[temp_1+1,temp_2]=255
            stack.append([temp_1+1,temp_2])
        if (a[2,2]<high_boundary) and (a[2,2] >low_boundary ):
            img_restrain[temp_1+1,temp_2+1]=255
            stack.append([temp_1+1,temp_2+1])
    for i in range(img_restrain.shape[0]):
        for j in range(img_restrain.shape[1]):
            if img_restrain[i,j]!=0 and img_restrain[i,j]!=255:
                img_restrain[i,j]=0
    plt.figure(4)
    plt.imshow(img_restrain.astype(np.uint8),cmap='gray')
    plt.axis('off')
    plt.show()




if __name__ == '__main__':
    canny1('lenna.png',5)