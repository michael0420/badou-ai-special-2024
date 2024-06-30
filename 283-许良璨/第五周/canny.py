import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path="../lenna.png"
    img=plt.imread(path)
    if path[-4:]==".png":
        img=img*255
    img=img.mean(axis=-1)
    plt.figure(1)
    plt.imshow(img.astype(np.uint8), cmap='gray')
    print("gray finish")

    sigma=0.5
    dim=5
    guassian_fliter=np.zeros([dim,dim])
    n1=1/(2*math.pi*sigma**2)
    n2=-1/(2*sigma**2)
    x=[dim-i-1 for i in range(dim)][::-1]
    for i in range(dim):
        for j in range(dim):
            guassian_fliter[i,j]=n1*math.exp(n2*(i**2+j**2))
    guassian_fliter=guassian_fliter/guassian_fliter.sum()

    img_pad=np.pad(img,((dim//2,dim//2),(dim//2,dim//2)),"constant")
    dy,dx=img.shape
    img_g=np.zeros([dy,dx])
    for i in range(dy):
        for j in range(dx):
            img_g[i,j]=np.sum(img_pad[i:i+dim,j:j+dim]*guassian_fliter)
    print("guassianlized finish")
    plt.figure(2)
    plt.imshow(img_g.astype(np.uint8), cmap="gray")




    sobel_x=np.array([[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]])
    sobel_y=np.array([[-1,-2,-1],
                      [0,0,0],
                      [1,2,1]])
    img_x=np.zeros([dy,dx])
    img_y=np.zeros([dy,dx])
    img_tidu=np.zeros([dy,dx])
    img_gpad=np.pad(img_g,((1,1),(1,1)),"constant")
    for i in range(dy):
        for j in range(dx):
            img_x[i,j]=np.sum(img_gpad[i:i+3,j:j+3]*sobel_x)
            img_y[i,j]=np.sum(img_gpad[i:i+3,j:j+3]*sobel_y)
            img_tidu[i,j]=np.sqrt(img_x[i,j]**2+img_y[i,j]**2)

    img_x[img_x==0]=0.00000001
    tan=img_y/img_x
    plt.figure(3)
    plt.imshow(img_tidu.astype(np.uint8),cmap="gray")
    print("sobel finish")

    img_yizhi=np.zeros([dy,dx])
    for i in range(1,dy-1):
        for j in range(1,dx-1):
            Flag=False
            
            temp=img_tidu[i-1:i+2,j-1:j+2]
            if tan[i,j]<-1:
                num1=(temp[0,1]-temp[0,0])/tan[i,j]+temp[0,1]
                num2=(temp[2,0]-temp[2,2])/tan[i,j]+temp[2,1]
                if img_tidu[i,j]>num1 and img_tidu[i,j]>num2:
                    Flag=True
            if tan[i,j]>1:
                num1=(temp[0,2]-temp[0,1])/tan[i,j]+temp[0,1]
                num2=(temp[2,0]-temp[2,1])/tan[i,j]+temp[2,1]
                if img_tidu[i,j]>num1 and img_tidu[i,j]>num2:
                    Flag=True
            if tan[i,j]<0:
                num1=(temp[1,0]-temp[0,0])*tan[i,j]+temp[1,0]
                num2=(temp[1,2]-temp[2,2])*tan[i,j]+temp[1,2]
                if img_tidu[i,j]>num1 and img_tidu[i,j]>num2:
                    Flag=True
            if tan[i,j]>0:
                num1=(temp[2,0]-temp[1,0])*tan[i,j]+temp[1,0]
                num2=(temp[0,2]-temp[1,2])*tan[i,j]+temp[1,2]
                if img_tidu[i,j]>num1 and img_tidu[i,j]>num2:
                    Flag=True
            if Flag:
                img_yizhi[i,j]=img_tidu[i,j]
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8),cmap="gray")
    print("non-max finish")

    low_boundary=img_tidu.mean()*0.5
    high_boundary=3*low_boundary
    zhan = []
    for i in range(1,dy-1):
        for j in range(1,dx-1):

            if img_yizhi[i,j]>=high_boundary:
                img_yizhi[i,j]=255
                zhan.append([i,j])
            elif img_yizhi[i,j]<=low_boundary:
                img_yizhi[i,j]=0
    print("before while")
    while not len(zhan) == 0:
        ty,tx=zhan.pop()
        temp=img_yizhi[ty-1:ty+2,tx-1:tx+2]
        if temp[0,0]!=0 and  temp[0,0]!=255:
            img_yizhi[ty-1,tx-1]=255
            zhan.append([ty-1,tx-1])
        if temp[0,1]!=0 and  temp[0,1]!=255:
            img_yizhi[ty-1,tx]=255
            zhan.append([ty-1,tx])
        if temp[0,2]!=0 and  temp[0,2]!=255:
            img_yizhi[ty-1,tx+1]=255
            zhan.append([ty-1,tx+1])
        if temp[1,0]!=0 and  temp[1,0]!=255:
            img_yizhi[ty,tx-1]=255
            zhan.append([ty,tx-1])
        if temp[1,2]!=0 and  temp[1,2]!=255:
            img_yizhi[ty,tx+1]=255
            zhan.append([ty,tx+1])
        if temp[2,0]!=0 and  temp[2,0]!=255:
            img_yizhi[ty+1,tx-1]=255
            zhan.append([ty+1,tx-1])
        if temp[2,1]!=0 and  temp[2,1]!=255:
            img_yizhi[ty+1,tx]=255
            zhan.append([ty+1,tx])
        if temp[2,2]!=0 and  temp[2,2]!=255:
            img_yizhi[ty+1,tx+1]=255
            zhan.append([ty+1,tx+1])
    print("while finish")

    for i in range(dy):
        for j in range(dx):
            if img_yizhi[i,j]!=255:
                img_yizhi[i,j]=0

    print("value check finish ")
    plt.figure(5)
    plt.imshow(img_yizhi.astype(np.uint8),cmap="gray")

    plt.show()
            

    

    
                    
                    
