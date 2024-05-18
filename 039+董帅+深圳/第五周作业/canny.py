import numpy as np
import math
import matplotlib.pyplot as plt
import cv2


def gaussian_blur(img,sigma=0.5,dim=5):
    gaussian_filter=np.zeros([dim,dim])
    distance=[i-dim//2 for i in range(dim)]
    n1=1/(2*math.pi*sigma**2)
    n2=-1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            gaussian_filter[i,j]=n1*math.exp((distance[i]**2+distance[j]**2)*n2)
    gaussian_filter=gaussian_filter/gaussian_filter.sum()
    dx,dy=img.shape
    img_new=np.zeros(img.shape)
    temp1=dim//2
    pad=np.pad(img,((temp1,temp1),(temp1,temp1)),'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i,j]=np.sum(pad[i:i+dim,j:j+dim]*gaussian_filter)
    return img_new.astype(np.uint8)

def compute_gradients(img):
    sobel_kernel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_kernel_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    img_tidu_x=np.zeros(img.shape)
    img_tidu_y=np.zeros(img.shape)
    img_tidu = np.zeros(img.shape)
    dx,dy=img.shape
    pad_tidu=np.pad(img,[[1,1],[1,1]],'constant')
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i,j]=np.sum(pad_tidu[i:i+3,j:j+3]*sobel_kernel_x)
            img_tidu_y[i,j]=np.sum(pad_tidu[i:i+3,j:j+3]*sobel_kernel_y)
            img_tidu[i,j]=np.sqrt(img_tidu_x[i,j] ** 2 + img_tidu_y[i,j] ** 2)
    img_tidu_x[img_tidu_x==0]=0.000000000001
    angle = img_tidu_y/img_tidu_x
    return img_tidu,angle

def non_maximum_suppression(img_tidu,angle):
    img_yizhi = np.zeros(img_tidu.shape)
    dx,dy = img_tidu.shape
    for i in range(1,dx-1):
        for j in range(1, dy-1):
            flag= True
            temp=img_tidu[i-1:i+2,j-1:j+2]#取八领域点，边缘点除外，所以从1开始，到y-1
            if angle[i,j] >= 1:#穿过1、3象限，y>x
                num1 = (temp[0,2] - temp[0,1])/angle[i,j] + temp[0,1]#线性插值公式
                num2 = (temp[2,0] - temp[2,1])/angle[i,j] + temp[2,1]
                if not (img_tidu[i,j] > num1 and img_tidu[i,j] > num2):#寻找像素点的局部最大值
                    flag = False
            elif angle[i,j] >0:#穿过1、3象限,y<x
                num1 = (temp[0,2] - temp[1,2])*angle[i,j] + temp[1,2]
                num2 = (temp[2,0] - temp[1,0])*angle[i,j] + temp[1,0]
                if not (img_tidu[i,j] > num1 and img_tidu[i,j] > num2):
                    flag = False
            elif angle[i,j] <=-1:
                num1 = (-temp[0,0] + temp[0,1])/angle[i,j] + temp[0,1]
                num2 = (-temp[2,2] + temp[2,1])/angle[i,j] + temp[2,1]
                if not (img_tidu[i,j] > num1 and img_tidu[i,j] > num2):
                    flag = False
            elif angle[i,j] <0:
                num1 = (temp[1,2] - temp[2,2])*angle[i,j] + temp[1,2]
                num2 = (temp[1,0] - temp[0,0])*angle[i,j] + temp[1,0]
                if not (img_tidu[i,j] > num1 and img_tidu[i,j] > num2):
                    flag = False
            if flag:
                img_yizhi[i,j] = img_tidu[i,j]
    return img_yizhi

def double_threshold(img_yizhi,lower_ratio=0.5, high_ratio=3.0):
    lower_boundary=img_tidu.mean()*lower_ratio
    high_boundary=lower_boundary*high_ratio
    zhan=[]
    dx,dy=img_yizhi.shape
    for i in range(1,dx-1):
        for j in range(1,dy-1):
            if img_yizhi[i,j] >= high_boundary:
                img_yizhi[i,j]=255
                zhan.append([i,j])
            elif img_yizhi[i,j] <= lower_boundary:
                img_yizhi[i,j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()
        a = img_yizhi[temp_1-1:temp_1+2,temp_2-1:temp_2+2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
    return img_yizhi

if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':
        img = img * 255
    img = img.mean(axis=-1)
    plt.figure(1)
    plt.imshow(img, cmap='gray')

    img_new = gaussian_blur(img)
    plt.figure(2)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')

    img_tidu, angle = compute_gradients(img_new)
    plt.figure(3)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    img_yizhi = non_maximum_suppression(img_tidu, angle)
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    img_yizhi_new=double_threshold(img_yizhi)
    plt.figure(5)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

