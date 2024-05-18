import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image

"""
灰度化：将RGB转为Gray

浮点算法：Gray = R0.3 + G0.59 +B0.11
整数算法：Gray = (R30 + G59 + B11)/100
移位方法：Gray = (R76 + G151 + B*28)>>8
平均值法：Gray = (R + G + B)/3
仅取绿色：Gray = G
"""
# 灰度化 ①
img = cv2.imread("lenna.png")       # cv2.imread()接口读图像，读进来直接是BGR 格式数据格式在 0~255
h,w = img.shape[:2]                 # 获取图片的high和wide
print("h：",h)
print("w：",w)
img_gray = np.zeros([h,w],img.dtype)    # 创建一张与原图片大小一样的单通道图片,img.dtype获取图像数据类型
# 获取每行每列各个点的RGB值 到 m
for i in range(h):
    for j in range(w):
        m = img[i,j]                # 将img图片第i行，第j列的RGB值赋值到m
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)        # 计算每个像素点的灰度值。！！！注意这里计算的顺序为 BGR
        # print("第",i,"行,第",j,"列的灰度值为",img_gray)
print("m",m)
print("img_gray",img_gray)
cv2.imshow("image show gray",img_gray)  # 固定写法
cv2.waitKey(3000)                       # 因闪退而添加的一行。  cv2.waitKey()参数是指等待的时间，默认单位为ms，为0时表示无限长时间

# 灰度化 ②
plt.subplot(221)                     # 表示将整个图像窗口分为2行2列, 当前位置为1
img2 = plt.imread("lenna.png")
plt.imshow(img2)                     # 展示原图
print("img2:",img2)
# plt.show()                           # 因闪退而添加的一行。

img_gray2 = rgb2gray(img2)           # 将真彩色图像 RGB 转换为灰度图像
plt.subplot(222)                     # 表示将整个图像窗口分为2行2列, 当前位置为2
plt.imshow(img_gray2, cmap='gray')   # 展示转换后的图，imshow固定写法
print("---image gray2----")
print(img_gray2)
# plt.show()                           # 因闪退而添加的一行。
"""  说明
cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
"""
# 二值化  只有两个值
img_two = np.where(img_gray2 >= 0.5,1,0)    # where(条件，符合条件得到的结果，不符合条件得到的结果)
print("img_two:",img_two)                   # 查看转换后的结果
print(img_two.shape)                        # 查看行数和列数

plt.subplot(223)                    # 表示将整个图像窗口分为2行2列, 当前位置为3
plt.imshow(img_two,cmap="gray")     # 展示二值化的结果，imshow固定写法
plt.show()


def chazhi(image):
    h,w,channels = image.shape      # 获取长宽和通道
    emptyImage = np.zeros((800,800,channels),np.uint8)        #创建1000×1000的图，与image通道一致
    Nh = 800/h                     # 倍率
    Nw = 800/w                      # 倍率
    for i in range(800):
        for j in range(800):
            x = int(i/Nh + 0.5)     # int 向下取整
            y = int(j/Nw + 0.5)
            emptyImage[i,j] = image[x,y]
    return emptyImage

image = cv2.imread("lenna.png")
zoom = chazhi(image)                # 调用函数，将执行的结果返回到zoom变量中
print("zoom",zoom)
print("zoom.shape",zoom.shape)
cv2.imshow("nearest interp",zoom)   # cv2.imshow固定写法
cv2.imshow("image",image)           # cv2.imshow固定写法
cv2.waitKey(0)