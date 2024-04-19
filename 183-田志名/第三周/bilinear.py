import numpy as np
import cv2

#双线性插值
def bilinear(img,newH,newW):
    preH,preW,c=img.shape
    newimage=np.zeros([newH,newW,c],dtype=np.uint8)   #np.uint8:0~255
    for i in range(c):
        for m in range(newH):
            for n in range(newW):
                #使两图像几何中心重合
                preX=(n+0.5)*(preW/newW)-0.5
                preY=(m+0.5)*(preH/newH)-0.5

                #确定四个点的位置
                preX0=max(0,int(np.floor(preX)))   #np.floor向下取整
                preX1=min(preX0+1,preW-1)
                preY0=max(0,int(np.floor(preY)))
                preY1=min(preY0+1,preH-1)

                #进行双线性插值
                R1=(preX1-preX)*img[preY0,preX0,i]+(preX-preX0)*img[preY0,preX1,i]
                R2=(preX1-preX)*img[preY1,preX0,i]+(preX-preX0)*img[preY1,preX1,i]
                newimage[m,n,i]=(int)((preY1-preY)*R1+(preY-preY0)*R2)
    return newimage

#最临近插值
def nearest(img,newH,newW):
    h, w, c = img.shape
    newimage = np.zeros([newH, newW, c], np.uint8)
    for i in range(newH):
        for j in range(newW):
            x = int(i * (h / newH) + 0.5)
            y = int(j * (w / newW) + 0.5)
            newimage[i, j] = img[x, y]
    return newimage


if __name__=='__main__':
    img=cv2.imread('lenna.png')
    newimage1 = bilinear(img, 700, 700)
    newimage2 = nearest(img, 700, 700)
    cv2.imshow("bilinear_img",newimage1)
    cv2.imshow("nearest_img", newimage2)
    cv2.waitKey(0)
