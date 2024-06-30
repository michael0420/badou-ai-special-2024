import numpy as np  # 导入NumPy库，用于处理数组
import cv2  # 导入OpenCV库，用于图像处理
from numpy import shape  # 从NumPy库中导入shape方法，用于获取数组的形状
import random  # 导入random库，用于生成随机数

def fun1(src,percetage):
	NoiseImg = src  # 将输入的图像赋值给NoiseImg变量
	NoiseNum = int(percetage * src.shape[0] * src.shape[1])  # 计算噪声点的数量

	for i in range(NoiseNum):  # 循环添加椒盐噪声
		# 每次取一个随机点
		# 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
		# random.randint生成随机整数
		# 椒盐噪声图片边缘不处理，故-1
		randX = random.randint(0, src.shape[0] - 1)  # 随机生成行坐标
		randY = random.randint(0, src.shape[1] - 1)  # 随机生成列坐标

	    # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
		if random.random() <= 0.5:  # 以50%的概率在该像素点设置为黑色（值为0）
			NoiseImg[randX, randY] = 0
		else:
			NoiseImg[randX, randY] = 255  # 以50%的概率在该像素点设置为白色（值为255）

	return NoiseImg  # 返回添加了椒盐噪声的图像数据

img = cv2.imread('lenna.png', 0)  # 读取灰度图像
# percent = 1来说就是二值图了，但是也不一定，可能还能看到点影子，但是就是
img1 = fun1(img, 0.2)  # 调用添加椒盐噪声的函数，生成带噪声的图像
#在文件夹中写入命名为lenna_PepperandSalt.png的加噪后的图片
#cv2.imwrite('lenna_PepperandSalt.png',img1)

# 显示原始灰度图像和添加椒盐噪声后的图像
img = cv2.imread('lenna.png')  # 以彩色模式读取图像
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像
cv2.imshow('source', img2)  # 显示原始灰度图像
cv2.imshow('lenna_PepperandSalt', img1)  # 显示添加椒盐噪声后的图像
cv2.waitKey(0)  # 等待用户关闭窗口