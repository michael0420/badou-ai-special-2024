'''
Python 实现灰度化，二值化
'''

from skimage.color import rgb2gray     #从skimage（Scikit-image）库的color模块中导入了rgb2gray函数
from skimage import io
import numpy as np
import matplotlib.pyplot as plt        #导入matplotlib库中的pyplot模块的标准方式，同时为它设置一个别名plt
from PIL import Image                  #导入Pillow库中的Image模块
import cv2                             #导入opencv库中的cv2模块

# 灰度化（编写函数灰度化）
img= io.imread("lenna.png")            #若用opencv读图，注意将RGB转换BGR
h,w=img.shape[:2]                      #获取图片的高度和宽度
img_gray=np.zeros([h,w],img.dtype)     #创建一个和原图相同数据类型的全黑灰度图像(单通道图像/数组中所有元素初始化为0 ）
for i in range(h):
for j in range(w):
m=img[i,j]                             #获取当前图像的high和wide坐标的BGR
img_gray[i,j]=int(m[0]*0.3+m[1]*0.59+m[2]*0.11)   #每个坐标的BGR转为gray并赋值给新图像

print('image show gray:%s'% img_gray)
cv2.imshow('image show gray',img_gray)            #显示灰度图
cv2.waitKey(0)使程序暂停

# 二值化（编写函数灰度化）
# 获取图片的高度和宽度
height,width=img_gray.shape
# 创建一个和原图相同数据类型的全黑灰度图像
img_binary=np.zeros([height,width],img_gray.dtype)
for i in range(height):
for j in range(width):
if img_gray[i,j]>=0.5:
img_binary[i, j]=1
else:
img_binary[i, j] = 0
cv2.imshow('image show binary',img_binary)
cv2.waitKey(0)

# 灰度化（调用函数灰度化）
plt.subplot(221)                                    #注意：cv2.imread读取图像plt.imshow()输出图像会失真
plt.imshow(img)
print("---image lenna----")

plt.subplot(222)
img_gray = rgb2gray(img)                            #调用rgb2gray()函数将RGB图形转换成灰度图像
img_gray=img.convert('L')                           #调用Image模块中convert()函数将RGB图形转换成灰度图像
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #调用cv2模块中cvt.Color函数将RGB图形转换成灰度图像
plt.imshow(img_gray,cmap='gray')
print("---image gray----")

# 二值化（调用函数二值化）
plt.subplot(223)
img_binary=np.where(img_gray>=0.5,1,0)               # plt读图，图像的每个像素值都在[0,1]之间
# cv2.threshold(img_binary,0.5,1,0,img_binary)       # cv2读图，图像的每个像素值都在[0.1]之间
print(img_binary.shape)                              #输出图像分辨率
plt.imshow(img_binary,cmap='gray')                   #在灰度化和二值化图像中用plt输出的时候必须添加cmap='gray'
print("---image binary----")
plt.show()