import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.color import rgb2gray
import cv2
if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    print("image", img)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255
    # 1 灰度化
    img_gray = rgb2gray(img)
    # 2 高斯滤波
    sigma = 0.5
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0.5)
    # 3 sobel算子
    dx, dy = img_gray.shape
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_sobel_x = np.zeros(img_blurred.shape)
    img_sobel_y = np.zeros(img_blurred.shape)
    img_sobel = np.zeros(img_blurred.shape)
    img_pad = np.pad(img_blurred, ((1,1), (1,1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_sobel_x[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)
            img_sobel_y[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)
            img_sobel[i,j] = np.sqrt(img_sobel_x[i,j]**2 + img_sobel_y[i,j]**2)
    img_sobel[img_sobel == 0] = 0.0000001
    angle = img_sobel_y/img_sobel_x
    # 4 极大值抑制
    img_restrain = np.zeros(img_sobel.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True
            near = img_sobel[i-1:i+2, j-1:j+2]
            if angle[i,j] <= -1:
                num1 = (near[0,1]-near[0,0]) / angle[i,j] + near[0,1]
                num2 = (near[2,1]-near[2,2]) / angle[i,j] + near[2,1]
                if not (img_sobel[i,j] > num1 and img_sobel[i,j] > num2):
                    flag = False
            elif angle[i,j] >= 1:
                num1 = (near[0,2]-near[0,1]) / angle[i,j] + near[0,1]
                num2 = (near[2,0]-near[2,1]) / angle[i,j] + near[2,1]
                if not (img_sobel[i,j] > num1 and img_sobel[i,j] > num2):
                    flag = False
            elif angle[i,j] > 0:
                num1 = (near[0,2]-near[1,2]) / angle[i,j] + near[1,2]
                num2 = (near[2,0]-near[1,0]) / angle[i,j] + near[1,0]
                if not (img_sobel[i,j] > num1 and img_sobel[i,j] > num2):
                    flag = False
            elif angle[i,j] < 0:
                num1 = (near[1,0]-near[0,0]) / angle[i,j] + near[1,0]
                num2 = (near[1,2]-near[2,2]) / angle[i,j] + near[1,2]
                if not (img_sobel[i,j] > num1 and img_sobel[i,j] > num2):
                    flag = False
            if flag:
                img_restrain[i,j] = img_sobel[i,j]
    # 5 双阈值检测
    lower = img_sobel.mean() * 0.9
    high = lower*3
    list = []  # 存放弱边缘坐标的列表
    for i in range(1,  dx-1):
        for j in range(1, dy-1):
            if img_restrain[i,j] > high:
                img_restrain[i,j] = 255
            elif img_restrain[i,j] < lower:
                img_restrain[i,j] = 0
            else:
                list.append([i,j])
    for x in list:
        tmp = img_restrain[x[0]-1:x[0]+2, x[1]-1:x[1]+2]  # 得到周围8个点
        #  标记周围是否有强边缘
        flag = False
        for value in np.nditer(tmp):
            #  周围有一个是强边缘
            if value > 255:
                flag = True
                break
        if flag:
            img_restrain[x[0],x[1]] = 255
        else:
            img_restrain[x[0], x[1]] = 0
    plt.imshow(img_restrain.astype(np.uint8), cmap='gray')
    plt.show()


