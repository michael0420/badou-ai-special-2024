import  cv2
import  numpy as np

img=cv2.imread('LYF.jpeg')
h,w,c=img.shape
#print(img.shape)
tar_h=400
tar_w=600
img2=np.zeros([tar_h,tar_w,c],dtype=np.uint8)
sc_h=float(h)/tar_h
sc_w=float(w)/tar_w
for i in range(c):
    for x in range(tar_h):
        for y in range(tar_w):
            #做中心重合
            x2=(x+0.5)*sc_w-0.5
            y2 = (y + 0.5) * sc_h - 0.5 
            #对x,y的取值做限制
            x0= min(int(np.floor(x2)),w-1) #这行代码计算了目标像素在水平方向上的左侧相邻像素的 x 坐标。np.floor() 函数返回不大于输入参数的最大整数，即向下取整，保证了该值为源图像中目标像素左侧相邻像素的 x 坐标。
            x1=min(x0+1,w - 1)
            y0 =  min(int(np.floor(y2)),h-1)
            y1 = min(y0 + 1, h - 1)
            #x方向做插值
            print(x2,y2,x0,y0,x1,y1)
            temp0=(x1-x2)*img[y0,x0,i]+(x2-x0)*img[y0,x1,i]
            temp1=(x1-x2)*img[y1,x0,i]+(x2-x0)*img[y1,x1,i]
            img2[x,y,i]=int((y1-y2)*temp0+(y2-y0)*temp1)

cv2.imshow('',img2)
cv2.waitKey()





