import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':

# 1.取灰度图
#    pic_path = 'lenna.png'
    img = plt.imread(r'E:\image\lenna.png')
    print("image", img)
    img = img * 255 
    img = img.mean(axis=-1)
    
#  2.高斯平滑 对图像进行降噪

    # sigma = 1 # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5# 高斯平滑时的高斯核参数，标准差，可调
    dim = 5 # 高斯核大小
    Gaussian_filter = np.zeros((dim, dim)) # 存储高斯核，这是数组
    tmp = [i - dim//2 for i in range(dim)] # 生成一个序列,为距离核中心（0，0）距离
    n1 = 1/(2*math.pi*sigma**2) # 计算高斯核
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx, dy = img.shape
    img_new = np.zeros((dx, dy)) # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim//2
    img_pad = np.pad(img, ((tmp,tmp),(tmp,tmp)), 'constant') #边缘填补
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * Gaussian_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')
#  3.计算图像梯度，得到可能边缘。
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0 ,1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2 ,-1]])
    img_tidu_x = np.zeros(img.shape)
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros([dx, dy])
    img_pad = np.pad(img, ((1,1),(1,1)), 'constant') #边缘填补
    for i in range(dx):
        j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_x) # x方向导数
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_y) # y方向导数
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
    img_tidu_x[img_tidu_x == 0] = 0.000000001
    angle = img_tidu_y / img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')
    

#  4.非极大值抑制，将有多个像素宽的边缘变成一个单像素宽的边缘。即“胖边缘”变成“瘦边缘”。    
#  局部范围内的梯度方向上，灰度变化最大的保留下来，其它的不保留
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i-1:i+2, j-1:j+2] # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1: # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_1 = (temp[2, 1] - temp[2, 0]) / angle[i, j] + temp[2, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')


#  5.双阈值检测，连接边缘。
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0]-1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1]-1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0
 
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1-1, temp_2-1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1-1, temp_2-1])  # 进栈
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
                
                
# 绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()
