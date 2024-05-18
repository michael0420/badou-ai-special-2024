import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    pic_path = "lenna.png"
    img = plt.imread(pic_path)  #.png 像素值在[0,1]
    print("image", img)
    if pic_path[-4:] == '.png': #.png 格式像素值*255
        img = img * 255
    #1.图像灰度化
    img = img.mean(axis=-1)    #取均值的方法进行灰度化
    #2.高斯滤波
    sigma = 0.5  #高斯平滑时的高斯核参数，标准差，可调
    dim = 5      #高斯核尺寸
    Gaussian_filter = np.zeros([dim,dim]) #生成dim*dim的高斯核，且值先为0
    tmp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma * sigma)
    n2 = 1 /(2 * sigma * sigma)

    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i,j] = n1 * math.exp(n2 * (tmp[i] * tmp[i] + tmp[j] * tmp[j]))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()

    dx, dy = img.shape
    img_new = np.zeros(img.shape) #存储平滑之后的图像，zeros函数得到的是浮点型数据
    r = dim // 2
    img_pad = np.pad(img, ((r,r), (r,r)), 'constant') # 边缘填补
    #img_pad[x:x+dim, y:y*dim] 从(0,0)开始，按照dim * dim的尺寸遍历图像
    for x in range(dx):
        for y in range(dy):
            img_new[x,y] = np.sum(img_pad[x:x+dim, y:y+dim] * Gaussian_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray') # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')

    #3.求梯度，以下使用sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    img_tidu_x = np.zeros(img.shape)
    img_tidu_y = np.zeros(img.shape)
    img_tidu = np.zeros(img.shape)
    img_pad = np.pad(img, ((1,1),(1,1)), 'constant')
    for x in range(dx):
        for y in range(dy):
            img_tidu_x[x,y] = np.sum(img_pad[x:x+3,y:y+3] * sobel_kernel_x)
            img_tidu_y[x,y] = np.sum(img_pad[x:x+3,y:y+3] * sobel_kernel_y)
            img_tidu[x,y] = np.sqrt(img_tidu_x[x,y] ** 2 + img_tidu_y[x,y] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x  #angle[x,y] = img_tidu_y[x,y] / img_tidu_x[x,y]
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    print(angle)
    #4.非极大值抑制
    img_nms = np.zeros(img_tidu.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
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
                img_nms[i, j] = img_tidu[i, j]


    plt.figure(3)
    plt.imshow(img_nms.astype(np.uint8), cmap='gray')
    plt.axis('off')

    #5. 双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3

    for x in range(1, img_nms.shape[0]-1):
        for y in range(1, img_nms.shape[1]-1):
            if img_nms[x,y] > high_boundary:
                img_nms[x,y] = 255
            elif img_nms[x,y] < lower_boundary:
                img_nms[x,y] = 0


    for x in range(1, img_nms.shape[0]-1):
        for y in range(1, img_nms.shape[1] - 1):
            if (img_nms[x,y] >= lower_boundary) and (img_nms[x,y] <= high_boundary):
                count = 0
                if img_nms[x-1, y-1] == 255:
                    count += 1
                if img_nms[x-1, y] == 255:
                    count += 1
                if img_nms[x-1, y+1] == 255:
                    count += 1
                if img_nms[x, y-1] == 255:
                    count += 1
                if img_nms[x, y+1] == 255:
                    count += 1
                if img_nms[x+1, y-1] == 255:
                    count += 1
                if img_nms[x+1, y] == 255:
                    count += 1
                if img_nms[x+1, y+1] == 255:
                    count += 1
                if count != 0:
                    img_nms[x,y] = 255
                    count = 0
                else:
                    img_nms[x, y] = 0

        # 绘图
        plt.figure(4)
        plt.imshow(img_nms.astype(np.uint8), cmap='gray')
        plt.axis('off')  # 关闭坐标刻度值




    plt.show()


