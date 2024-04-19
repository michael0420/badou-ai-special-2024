'''引入opencv,numpy,matplotlib等库'''
'''
@AUTHER:ZENOWANG
彩色图像灰度化
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt  # 也可用import matplotlib.pyplot as plt

'''手动转化为灰度图gray_figure'''
figure = cv2.imread('lenna.png')
print(figure.dtype, figure.shape, figure.size, figure[0][0])
h, w, c = figure.shape
gray_figure = np.ones((h, w), np.uint8)
# 将每个元素位置的三通道信息转化为单通道灰度
for i in range(h):
    for j in range(w):
        pixel = figure[i][j]
        gray_figure[i][j] = int(pixel[0] * 0.11 + pixel[1] * 0.59 + pixel[2] * 0.30 + 0.5)  # int向下取整，+0.5后更接近float值
print(gray_figure)
cv2.imshow('demo1', gray_figure)  # 将gray_figure矩阵以图像显示，名字为demo1
cv2.waitKey(1000)  # 显示3000ms后关闭

'''对比显示，原图显示在241位置，用cv2读入，用plt输出'''
plt.subplot(2, 4, 1)
img_0 = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
plt.imshow(img_0)  # 这里前面用cv2读入图像格BGR，如果用plt.imshow输出为RGB格式，需在之前进行cv2.cvtCOLOR(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img_gray, cmap='jet')  # cmap参数只对BRG2GRAY有效，且可以修改表现形式
'''使用接口转换灰度图显示在245-8位置，分别显示不同的颜色样式'''
img = cv2.imread('lenna.png')
# img_gray = rgb2gray(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(2, 4, 5)
# viridis默认的颜色映射，从深蓝到亮黄；gray灰度颜色映射；hot热量图颜色映射，显示温度分布；jet彩虹颜色映射。不受plt或cv2读入方式影响
plt.imshow(img_gray)  # 或用cmap=viridis
# print(img_gray)
plt.subplot(2, 4, 6)  # 不使用此命令，生成图像会覆盖掉3，5，10位置即上一个图
plt.imshow(img_gray, cmap='gray')
plt.subplot(2, 4, 7)
plt.imshow(img_gray, cmap='hot')
plt.subplot(2, 4, 8)
plt.imshow(img_gray, cmap='jet')

plt.show()





# cv2.COLOR_BRT2GRAY的原理逻辑是什么，既然已经to gray，为何imshow还可以用cmap控制？？？















'''
# cv2.imread 读进来都是uint8格式
# plt.imread(path) 根据图像本身自动判断读进来是灰度还是RGB: 
# plt本身是用来显示图片的,不建议读图片,如果非要用plt读图片, 则读.png是float32格式, 读其他格式进来时uint8

# info = cv2.imread('lenna.png')  # 图片格式化储存矩阵到info
# h, w = info.shape[0:2]  # 根据需要转化某些维度信息
# channels = info.shape[2]
# # h, w, channels = info.shape  # shape函数转化三个维度信息
# print(h, w, channels)  # 输出图片长、宽、通道数
# print(info.size)  # 输出图片所有元素，长×宽×通道数 = 512 * 512 * 3
# # print(info[0])  # 输出第一行元素的通道信息，是BGR格式
# # print(info[0][0])  # 输出第一个位置元素的通道信息，125,137,226，是BGR格式
#
# # 新建形状与原图h,w相同的单通道图片，位置类型是uint8=2^8=256，u表示非负，0~255;uint8=2^8=256，-128~127;uint32=2^32
# float_gray = np.zeros([h, w], dtype=np.uint8)  #dtype=np.uint8或np.uint8或dtype=info.dtype或info.dtype
# int_gray = np.zeros([h, w], dtype=np.uint8)
# ave_gray = np.zeros([h, w], dtype=np.uint8)
# for i in range(h):
#     for j in range(w):
#         m = info[i][j]
#         float_gray[i][j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
#         int_gray[i][j] = int(m[0]*11 + m[1]*59 + m[2]*30)/100
#         ave_gray[i][j] = int((m[0] + m[1] + m[2]) / 3)
# # print(info.dtype)
# # print(new_img.dtype)
# # print(new_img[0][0])  # 转化灰度图后第一个元素值
# # print(m)
# # print(new_img.shape)  # 没有shape[2]维度
# # print('output gray picture is %s' % 'info')
# # cv2.imshow('Demo1', float_gray)  # 将矩阵转化为图片输出，图片框名字为Demo1
# # cv2.waitKey(3000)  # 等待时间，ms毫秒，按任意键退出
# # 创建一个新的Figure对象，并在其中创建一个或多个子图，方便进行对比，用plt.subplot()
'''
