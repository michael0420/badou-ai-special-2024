import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    # print('image',img)
    if pic_path[-4:] == '.png':            #.png图片在这里的存储格式是0-1的浮点数。所以要
        img = img * 255    #还是浮点数的类型
    # print("255",img)
    # 对图像进行灰度化
    img = img.mean(axis=-1)  #axis=-1表示最后一个轴,在最后一个轴取均值即灰度化,最后一个轴是3通道
    # print('gray',img)

# 1：高斯平滑
# sigma = 0.5 高斯平滑时的高斯核函数，标准差，可调
    sigma = 0.5
    dim = 5
    Gaussian_filter = np.zeros([dim,dim])
    # print(Gaussian_filter)
    tmp = [i-dim//2 for i in range(dim)]
    # print("tmp:\n",tmp)
    n1 = 1/(2*math.pi*sigma**2)
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i,j]=n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    # print(Gaussian_filter)
    Gaussian_filter = Gaussian_filter/Gaussian_filter.sum()
    # print(Gaussian_filter)
    dx, dy = img.shape
    img_new = np.zeros(img.shape) #用来存储平滑之后的图像，zeros函数得到的浮点型数据
    tmp= dim//2    #数值是2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)),"constant")  #边缘填补
    # print(img_pad)
    for i in range(dx):
        for j in range(dy):
            img_new[i,j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 2:求梯度，用sobel 矩阵（检测图像中的水平,垂直和对焦边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)
    img_tidu_y = np.zeros(img_new.shape)
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new,((1, 1),(1, 1)),"constant")
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)
            img_tidu[i,j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    # print("img_tidu_x:\n", img_tidu_x)
    # print("img_tidu_y:\n", img_tidu_y)
    angle = img_tidu_y/img_tidu_x
    print("....angle.....\n",angle)
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8),cmap='gray')
    plt.axis('off')

    #3:非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1,dx-1):
        for j in range(1,dy-1):
            flag = True   #在8领域内是否要抹去
            temp = img_tidu[i-1:i+2,j-1:j+2]  #在8领域j矩阵
            if angle[i, j] <=-1:
                num_1 = (temp[0, 1]-temp[0, 0])/angle[i, j]+temp[0, 1]
                num_2 = (temp[2, 1]-temp[2, 2])/angle[i, j]+temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
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
    print("极大值抑制之后像素点：\n",img_yizhi[224:228,224:226])
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    #双阈值检测，连接边缘，遍历所有一定是边的点，查看8领域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean()
    high_boundary = lower_boundary * 3
    print(lower_boundary,high_boundary)
    zhan = []
    for i in range(1, img_yizhi.shape[0]-1):
        for j in range(1, img_yizhi.shape[1]-1):
            if img_yizhi[i, j] >= high_boundary:
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:
                img_yizhi[i, j] = 0
    print(zhan)
    print(len(zhan))
    while not len(zhan) == 0:
        temp_1,temp_2 = zhan.pop()
        a = img_yizhi[temp_1-1:temp_1+2,temp_2-1:temp_2+2]
        #强边缘周围8领域，如果有弱边缘的值，直接变成强边缘
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255
            zhan.append([temp_1 - 1, temp_2 - 1])
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
            if img_yizhi[i, j] !=0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
    # 绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()