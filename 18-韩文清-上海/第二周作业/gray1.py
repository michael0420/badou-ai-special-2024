from skimage.color import rgb2gray   #导入scikit-image 库中的color模块，并从中导入了rgb2gray函数，将 RGB 彩色图像转换为灰度图像。
import numpy as np  #将numpy库命名为np
import matplotlib.pyplot as plt  #将matplotlib.pyplot库命名为plt
from PIL import Image  #Python Imaging Library（PIL）中导入 Image 模块。Python Imaging Library（PIL）是Python中用于处理图像的库，Image模块提供了图像处理的核心功能。
import cv2

#灰度化
img=cv2.imread("lenna.png")
h,w=img.shape[:2]  #获取图片的高和宽
img_gray = np.zeros([h,w],img.dtype)  #创建high为h，wide为w的全零数组，mg.dtype表示新数组的数据类型与原始图像img的数据类型相同
for i in range(h):
    for j in range(w):
        img_gray[i,j] = 0.3*img[i,j][0]+0.59*img[i,j][1]+0.11*img[i, j][2]    ##将BGR坐标转化为gray坐标并赋值给新图像
print(img_gray)
print("image show gray: %s"%img_gray)  #%s是占位符，用于插入img_gray的值
#cv2.imshow("image show gray",img_gray) #创建一个名为 "image show gray" 的窗口，并在该窗口中显示灰度图像img_gray
plt.imshow(img_gray, cmap='gray') #使用 Matplotlib 来显示图像
plt.title('image show gray') #将图片标题设置为 "image show gray"
plt.show()
