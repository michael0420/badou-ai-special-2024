import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

#灰度化
#1、手工实现
img = cv2.imread('1.png')    #读入图片，绝对路径或相对路径
height, width, channels = img.shape
img_gray = np.zeros([height, width], img.dtype)  #使用np.zeros生成宽、高与图片像素一致的全部是0的矩阵，准备存储生成后的值
for i in range(height):                          #遍历img矩阵，取出每个像素的BRG值
    for j in range(width):
        img_gray[i, j] = int(0.11 * img[i, j][0] + 0.59 * img[i, j][1] + 0.3 * img[i, j][2])   #Gray = 0.3R+0.59G+0.11B，cv2读取的顺序是BGR
cv2.imwrite('img_gray.png', img_gray)    #保存

#2、调用cv接口
img = cv2.imread('1.png')
img_gray_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #直接调用接口即可，cv2.cvtColor，注意cv2读取的是BGR顺序
cv2.imwrite('img_gray_2.png', img_gray_2)

#3、调用skimage的rgb2gray接口
img = plt.imread('1.png')    #plt读取之后对矩阵做了归一化，就是所有值都除以255，读取的顺序是RGB
img_gray_3 = rgb2gray(img)   #from skimage.color import rgb2gray
plt.imsave('img_gray_3.png', img_gray_3, cmap='gray')

#二值化
#1、手工实现-cv2
img = cv2.imread('1.png')
height, width = img.shape[:2]
img_gray = np.zeros([height, width], img.dtype)      #通过灰度化实现二值化
img_binary = np.zeros([height, width], img.dtype)    #构建初始矩阵
for i in range(height):
    for j in range(width):
        img_gray[i, j] = int(0.11 * img[i, j][0] + 0.59 * img[i, j][1] + 0.3 * img[i, j][2])   #灰度化
        img_bin = img_gray[i, j]/255    #归一化
        if img_bin >= 0.5:
            img_binary[i, j] = 255      #二值赋值
        else:
            img_binary[i, j] = 0
cv2.imwrite('img_binary.png', img_binary)

#2、手工实现-plt
img = plt.imread('1.png')
height, width = img.shape[0:2]
img_binary_2 = np.zeros([height, width], img.dtype)
for i in range(height):
    for j in range(width):
        if img_gray_3[i, j] >= 0.5:          #使用上面已经灰度化的img_gray_3，由于已经归一化，所以直接使用即可
            img_binary_2[i, j] = 1
        else:
            img_binary_2[i, j] = 0
plt.imsave('img_binary_2.png', img_binary_2, cmap='gray')

#3.使用numpy
img = plt.imread('1.png')
img_gray = rgb2gray(img)
img_binary_3 = np.where(img_gray >= 0.5, 1, 0)      #书写简单
plt.imsave('img_binary_3.png', img_binary_3, cmap='gray')















