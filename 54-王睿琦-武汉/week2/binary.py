import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('liuyifei.jpg') #读图
image2 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
image = cv2.resize(image2, (720,1280)) #图片缩放

imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #将BGR格式转换成灰度图片
h, w = image.shape[0:2] #取出图像的高度和宽度
imageBinary = np.zeros([h, w], imageGray.dtype) #创建一张和当前灰度图大小一样的单通道图片

for x in range(h):
    for y in range(w):
        if imageGray[x,y] / 255 < 0.5:
            imageBinary[x,y] = 0
        else:
            imageBinary[x,y] = 255
#实现二值化

#方法一
# cv2.imshow("yuantu",image) #输出原图
# cv2.imshow("erzhihuahou",imageBinary) #输出二值化后的图
# cv2.waitKey() #等待用户按键输入无限等待
# cv2.destroyAllWindows() #销毁我们创建的所有窗口

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(imageBinary, cmap='gray')
plt.show()
