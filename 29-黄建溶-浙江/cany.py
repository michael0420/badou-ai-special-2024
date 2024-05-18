import numpy as np
import matplotlib.pyplot as plt
import math
if __name__ == '__main__':
 img_path='lenna.png'
 img=plt.imread(img_path)     #　读图
 print('img_1',img)
 if img_path[-4:]=='.png':# .png格式的图片的存储值为0和1的浮点数，转化为255
     img=img*225
 img=img.mean(axis=-1)  # 利用取均值的方法进行灰度化

# 高斯平滑
 sigma=1.52  # 高斯平滑的参数设置
 dim=5       # 卷积核的维度设置
 Gaussian_filter=np.zeros([dim,dim]) # 存储高斯核，这是数组不是列表了
 tmp=[i-dim//2 for i in range(dim)]  # 生成一个[-2,2]的列表
 n1=1/(2*math.pi*sigma**2)
 n2=-1/(2*sigma**2)
 for x in range(dim):
     for y in range(dim):
         Gaussian_filter[x,y]=n1*math.exp(n2*(tmp[x]**2+tmp[y]**2))     # 计算卷积核
 Gaussian_filter=Gaussian_filter/Gaussian_filter.sum()                  # 得到卷积核
 dx,dy=img.shape
 img_new=np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
 tmp=dim//2
 img_pad=np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补
 for i in range(dx):
     for j in range(dy):
         img_new[i,j]=np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)         # 对图像各点进行遍历，高斯平滑
 plt.figure(1)
 plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，再强制转换，gray灰阶
 plt.axis('off')
 # 求梯度
 kernel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
 kernel_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
 tidu_x=np.zeros(img_new.shape)
 tidu_y=np.zeros([dx,dy])
 tidu_image=np.zeros(img_new.shape)
 img_pad=np.pad(img,((1,1),(1,1)),'constant')  # 对图像进行边缘填充，填充数值为常数，大小为一个像素点
 for i in range(dx):
  for j in range(dy):
   tidu_x[i,j]=np.sum(img_pad[i:i+3,j:j+3]*kernel_x)  # 局部卷积算x方向的梯度
   tidu_y[i,j]=np.sum(img_pad[i:i+3,j:j+3]*kernel_y)  # 局部卷积算y方向的梯度
   tidu_image[i,j]=np.sqrt(tidu_x[i,j]**2+tidu_y[i,j]**2)
 tidu_x[tidu_x==0]=0.0000001
 tidu_toward=tidu_y/tidu_x    #
 plt.figure(2)
 plt.imshow(tidu_image.astype(np.uint8),cmap='gray')
 plt.axis('off')

 # 非极大值抑制
 yizhi_image=np.zeros(tidu_image.shape)
 for i in range(1,dx-1):
  for j in range(1,dy-1):
   flag=True              # 判断标志位
   temp=tidu_image[i-1:i+2,j-1:j+2]    # 邻近点
   if tidu_toward[i,j]<=-1:            # 判断方向
    num_1=(temp[0,1]-temp[0,0])/tidu_toward[i,j]+temp[0,1]     # 插值运算，并判断大小
    num_2=(temp[2,1]-temp[2,2])/tidu_toward[i,j]+temp[2,1]
    if not (tidu_image[i,j]>num_1 and tidu_image[i,j]>num_2):
     flag=False
   elif tidu_toward[i,j]>= 1:
    num_1=(temp[0,2]-temp[0,1])/tidu_toward[i,j]+temp[0,1]
    num_2=(temp[2,0]-temp[2,1])/tidu_toward[i,j]+temp[2,1]
    if not (tidu_image[i,j]>num_1 and tidu_image[i,j]>num_2):
     flag=False
   elif tidu_toward[i,j]>0:
    num_1=(temp[0,2]-temp[1,2])*tidu_toward[i,j]+temp[1,2]
    num_2=(temp[2,0]-temp[1,0])*tidu_toward[i,j]+temp[1,0]
    if not (tidu_image[i,j]>num_1 and tidu_image[i,j]>num_2):
     flag=False
   elif tidu_toward[i,j]<0:
    num_1=(temp[1,0]-temp[0,0])*tidu_toward[i,j]+temp[1,0]
    num_2=(temp[1,2]-temp[2,2])*tidu_toward[i,j]+temp[1,2]
    if not (tidu_image[i,j]>num_1 and tidu_image[i,j]>num_2):
     flag=False

   if flag==True:
    yizhi_image[i,j]=tidu_image[i,j]
 plt.figure(3)
 plt.imshow(yizhi_image.astype(np.uint8), cmap='gray')
 plt.axis('off')

  # 双阈值检测
 low_yuzhi=tidu_image.mean() * 0.5    # 设置低阈值
 high_yuzhi=low_yuzhi*3               # 设置高阈值
 zhan = []
 for i in range(1,yizhi_image.shape[0]-1):      # 遍历图像的点，进行判断
  for j in range(1,yizhi_image.shape[1]-1):
   if yizhi_image[i,j] >= high_yuzhi:
    yizhi_image[i,j]=255
    zhan.append([i, j])
   if yizhi_image[i,j] <= low_yuzhi:
    yizhi_image[i,j]=0

 while not len(zhan) == 0:
  temp_1, temp_2 = zhan.pop()  # 出栈
  a = yizhi_image[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
  if (a[0, 0] < high_yuzhi) and (a[0, 0] > low_yuzhi):
   yizhi_image[temp_1 - 1, temp_2 - 1] = 255          # 这个像素点标记为边缘
   zhan.append([temp_1 - 1, temp_2 - 1])           # 进栈
  if (a[0, 1] < high_yuzhi) and (a[0, 1] > low_yuzhi):
   yizhi_image[temp_1 - 1, temp_2] = 255
   zhan.append([temp_1 - 1, temp_2])
  if (a[0, 2] <high_yuzhi) and (a[0, 2] > low_yuzhi):
   yizhi_image[temp_1 - 1, temp_2 + 1] = 255
   zhan.append([temp_1 - 1, temp_2 + 1])
  if (a[1, 0] < high_yuzhi) and (a[1, 0] > low_yuzhi):
   yizhi_image[temp_1, temp_2 - 1] = 255
   zhan.append([temp_1, temp_2 - 1])
  if (a[1, 2] < high_yuzhi) and (a[1, 2] > low_yuzhi):
   yizhi_image[temp_1, temp_2 + 1] = 255
   zhan.append([temp_1, temp_2 + 1])
  if (a[2, 0] < high_yuzhi) and (a[2, 0] >low_yuzhi):
   yizhi_image[temp_1 + 1, temp_2 - 1] = 255
   zhan.append([temp_1 + 1, temp_2 - 1])
  if (a[2, 1] < high_yuzhi) and (a[2, 1] > low_yuzhi):
   yizhi_image[temp_1 + 1, temp_2] = 255
   zhan.append([temp_1 + 1, temp_2])
  if (a[2, 2] < high_yuzhi) and (a[2, 2] > low_yuzhi):
   yizhi_image[temp_1 + 1, temp_2 + 1] = 255
   zhan.append([temp_1 + 1, temp_2 + 1])

 for i in range(yizhi_image.shape[0]):
  for j in range(yizhi_image.shape[1]):
   if yizhi_image[i, j] != 0 and yizhi_image[i, j] != 255:
    yizhi_image[i, j] = 0

 plt.figure(4)
 plt.imshow(yizhi_image.astype(np.uint8), cmap='gray')
 plt.axis('off')
 plt.show()