import numpy as np
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


#灰度化
def gray(img):
    h,w=img.shape[:2]
    img_gray = np.zeros([h,w],img.dtype) #dtype数据类型
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            #浮点整数算法,cv读入是BGR
            img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
    return img_gray


img = cv2.imread("lenna.png")

#未进行归一化处理
#img_gray = gray(img)
#二值
#img_binary = np.where(img_gray >  128, 1, 0)

#进行了归一化处理的灰度化
img_gray = rgb2gray(img)
#二值
img_binary = np.where(img_gray >= 0.5,1,0)
plt.subplot(131) #一行三列第一个
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #cv2显示是brg，要用plt展示的话需要转换成rgb
plt.title('Original Image')

plt.subplot(132)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(133)
plt.imshow(img_binary,cmap='gray')
plt.title('Binary Image')

plt.show()  # 显示图像



cv2.waitKey(0)