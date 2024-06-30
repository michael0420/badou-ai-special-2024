import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

#图像进行灰度化
img = cv.imread("lenna.png", flags=0)
# print(img.shape)

#图像进行高斯滤波
sigma = 0.5
dim = 5
gaussFilter = np.zeros([dim, dim])
tmp = [i - dim//2 for i in range(dim)]
n1 = 1/(2 * math.pi * sigma ** 2)
n2 = -1/(2 * sigma ** 2)
for i in range(dim):
    for j in range(dim):
        gaussFilter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))

gaussFilter = gaussFilter/gaussFilter.sum()

h, w = img.shape
imageNew = np.zeros(img.shape)
tmp = dim//2
img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
for i in range(h):
    for j in range(w):
        imageNew[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * gaussFilter)

plt.figure(1)
plt.imshow(imageNew.astype(np.uint8), cmap='gray')
plt.axis('off')

#检查图像中的水平、垂直和对角边缘
sobelx = np.array([[-1, 0 ,1], [-2, 0, 2],[-1, 0 ,1]])
sobely = np.array([[1, 2 ,1], [0, 0, 0],[-1, -2 ,-1]])
imgTiduX =  np.zeros(imageNew.shape)
imgTiduY = np.zeros(imageNew.shape)
imgTidu  = np.zeros(imageNew.shape)
img_pad = np.pad(imageNew, ((1, 1), (1, 1)), 'constant')

for i in range(h):
    for j in range(w):
        imgTiduX[i ,j] = np.sum(img_pad[i:i+3, j:j+3] * sobelx)
        imgTiduY[i ,j]= np.sum(img_pad[i:i + 3, j:j + 3] * sobely)
        imgTidu[i ,j]  = np.sqrt(imgTiduX[i, j]**2 + imgTiduY[i, j]**2)

imgTiduX[imgTiduX == 0] = 0.0000001
angle = imgTiduY/imgTiduX
plt.figure(2)
plt.imshow(imgTidu.astype(np.uint8), cmap='gray')
plt.axis('off')

#对梯度幅值进行非极大值抑制
imgYizhi = np.zeros(imgTidu.shape)
#分别判断梯度上的两个点和邻近点的大小,8个点，以正负一为界限进行判断
for i in range(1, h - 1):
    for j in range(1, w - 1):
        flag = True
        temp = imgTidu[i-1:i+2, j-1:j+2]
        if angle[i, j] <= -1:
            num1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (imgTidu[i, j] > num1 and imgTidu[i, j] > num2):
                flag = False
        elif angle[i, j] >= 1:
                num1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (imgTidu[i, j] > num1 and imgTidu[i, j] > num2):
                    flag = False
        elif angle[i, j] > 0:
                num1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (imgTidu[i, j] > num1 and imgTidu[i, j] > num2):
                    flag = False
        elif angle[i, j] < 0:
                num1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (imgTidu[i, j] > num1 and imgTidu[i, j] > num2):
                    flag = False
        if flag:
            imgYizhi[i, j] = imgTidu[i, j]

plt.figure(3)
plt.imshow(imgYizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

#用双阈值算法检测和连接边缘
lowerLimit = imgTidu.mean() * 0.5
highLimit = imgTidu.mean() * 1.5
stack = []
for i in range(1, imgYizhi.shape[0] - 1):
    for j in range(1, imgYizhi.shape[1] - 1):
        if(imgYizhi[i, j] >= highLimit):
            imgYizhi[i, j] = 255
            stack.append([i,j])
        elif imgYizhi[i, j] <= lowerLimit:
            imgYizhi[i, j] = 0

while not len(stack) == 0 :
    temp1, temp2 = stack.pop()
    a = imgYizhi[temp1 - 1 : temp1 + 2, temp2 - 1 : temp2 + 2]
    if(a[0, 0] < highLimit and a[0, 0] > lowerLimit):
        imgYizhi[temp1 - 1, temp2 - 1] = 255
        stack.append([temp1 - 1, temp2 - 1])
    elif (a[0, 1] < highLimit and a[0, 1] > lowerLimit):
        imgYizhi[temp1 - 1, temp2] = 255
        stack.append([temp1 - 1, temp2])
    elif (a[0, 2] < highLimit and a[0, 2] > lowerLimit):
        imgYizhi[temp1 - 1, temp2 + 1] = 255
        stack.append([temp1 - 1, temp2 + 1])
    elif (a[1, 0] < highLimit and a[1, 0] > lowerLimit):
        imgYizhi[temp1, temp2 - 1] = 255
        stack.append([temp1, temp2 - 1])
    elif (a[1, 2] < highLimit and a[1, 2] > lowerLimit):
        imgYizhi[temp1, temp2 + 1] = 255
        stack.append([temp1, temp2 + 1])
    elif (a[2, 0] < highLimit and a[2, 0] > lowerLimit):
        imgYizhi[temp1 + 1, temp2 - 1] = 255
        stack.append([temp1 + 1, temp2 - 1])
    elif (a[2, 1] < highLimit and a[2, 1] > lowerLimit):
        imgYizhi[temp1 + 1, temp2] = 255
        stack.append([temp1 + 1, temp2])
    elif (a[2, 2] < highLimit and a[2, 2] > lowerLimit):
        imgYizhi[temp1 + 1, temp2 + 1] = 255
        stack.append([temp1 + 1, temp2 + 1])

for i in range(imgYizhi.shape[0]):
    for j in range(imgYizhi.shape[1]):
        if(imgYizhi[i, j] != 0 and imgYizhi[i, j] != 255):
            imgYizhi[i, j] = 0

#绘图
plt.figure(4)
plt.imshow(imgYizhi.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.show()